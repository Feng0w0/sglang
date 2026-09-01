"""Opt-in hidden-state traces for SGLang teacher-forced prefill.

The trace records the hidden vector that predicts the first response token at
stable transformer boundaries.  Events are correlated with Megatron by the
SHA256 of the complete prompt+response token sequence.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
from collections import defaultdict

import torch


_MARKER = "GLM52_DFX_PREFILL_HIDDEN="
_EMITTED: defaultdict[str, int] = defaultdict(int)


def _enabled() -> bool:
    return os.environ.get("GLM52_DFX_PREFILL") == "1"


def should_trace_layer_detail(layer_id: int) -> bool:
    """Return whether stable internal boundaries should be traced for a layer."""

    if not _enabled() or os.environ.get("GLM52_DFX_LAYER_DETAIL") != "1":
        return False
    selected = os.environ.get("GLM52_DFX_LAYER_DETAIL_LAYERS", "0").strip().lower()
    if selected in {"all", "*"}:
        return True
    for item in selected.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            start, end = item.split("-", 1)
            if int(start) <= layer_id <= int(end):
                return True
        elif int(item) == layer_id:
            return True
    return False


def should_trace_attention_detail(layer_id: int) -> bool:
    """Return whether TP-local attention intermediates should be traced."""

    if not should_trace_layer_detail(layer_id):
        return False
    if os.environ.get("GLM52_DFX_ATTN_DETAIL") != "1":
        return False
    selected = os.environ.get(
        "GLM52_DFX_ATTN_DETAIL_LAYERS",
        os.environ.get("GLM52_DFX_LAYER_DETAIL_LAYERS", "0"),
    ).strip().lower()
    if selected in {"all", "*"}:
        return True
    for item in selected.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            start, end = item.split("-", 1)
            if int(start) <= layer_id <= int(end):
                return True
        elif int(item) == layer_id:
            return True
    return False


def _tp_rank() -> int:
    try:
        from sglang.srt.distributed.parallel_state import (
            get_tensor_model_parallel_rank,
        )

        return get_tensor_model_parallel_rank()
    except Exception:
        return int(os.environ.get("RANK", "0"))


def _tp_size() -> int:
    try:
        from sglang.srt.distributed.parallel_state import (
            get_tensor_model_parallel_world_size,
        )

        return get_tensor_model_parallel_world_size()
    except Exception:
        return int(os.environ.get("WORLD_SIZE", "1"))


def _sequence_slices(input_ids: torch.Tensor, forward_batch):
    """Yield uncached, teacher-forced sequences in the flattened prefill batch."""

    if not forward_batch.forward_mode.is_extend():
        return
    if not forward_batch.is_prefill_only or not forward_batch.return_logprob:
        return

    lengths = forward_batch.extend_seq_lens_cpu
    if lengths is None and forward_batch.extend_seq_lens is not None:
        lengths = forward_batch.extend_seq_lens.detach().cpu().tolist()
    prefixes = forward_batch.extend_prefix_lens_cpu
    if prefixes is None:
        prefixes = [0] * len(lengths or [])
    if not lengths or len(lengths) != len(prefixes):
        return

    offset = 0
    for sequence_index, (length, prefix_length) in enumerate(zip(lengths, prefixes)):
        length = int(length)
        prefix_length = int(prefix_length)
        # A non-zero prefix means input_ids contains only an incremental suffix,
        # so it cannot be hashed against Megatron's complete sequence.  The GLM
        # consistency launcher disables the radix cache to keep this at zero.
        if prefix_length == 0 and length > 0 and offset + length <= input_ids.numel():
            yield sequence_index, offset, length
        offset += length


@torch.no_grad()
def trace_sglang_prefill_hidden(
    boundary: str,
    hidden_states: torch.Tensor,
    input_ids: torch.Tensor | None,
    forward_batch,
) -> None:
    """Emit the prediction-position vector for each teacher-forced sequence."""

    if not _enabled() or _tp_rank() != 0:
        return
    if input_ids is None or not isinstance(hidden_states, torch.Tensor):
        return

    max_calls = max(1, int(os.environ.get("GLM52_DFX_PREFILL_MAX_CALLS", "4")))
    response_length = max(
        1, int(os.environ.get("GLM52_DFX_PREFILL_RESPONSE_LENGTH", "1"))
    )
    if _EMITTED[boundary] >= max_calls:
        return

    flat_ids = input_ids.detach().reshape(-1)
    for sequence_index, offset, total_length in _sequence_slices(
        flat_ids, forward_batch
    ) or ():
        if total_length <= response_length or _EMITTED[boundary] >= max_calls:
            continue
        prediction_position = total_length - response_length - 1
        tensor_position = offset + prediction_position
        if tensor_position >= hidden_states.shape[0]:
            continue

        sequence = flat_ids[offset : offset + total_length].to(
            device="cpu", dtype=torch.int64
        )
        vector = (
            hidden_states[tensor_position]
            .detach()
            .reshape(-1)
            .float()
            .cpu()
            .contiguous()
        )
        vector_bytes = vector.numpy().tobytes()
        event = {
            "backend": "sglang",
            "boundary": boundary,
            "sequence_index": sequence_index,
            "token_sha256": hashlib.sha256(sequence.numpy().tobytes()).hexdigest(),
            "total_length": total_length,
            "response_length": response_length,
            "prediction_position": prediction_position,
            "response_token": int(sequence[-response_length].item()),
            "hidden_size": vector.numel(),
            "vector_dtype": "float32",
            "vector_sha256": hashlib.sha256(vector_bytes).hexdigest(),
            "vector_b64": base64.b64encode(vector_bytes).decode("ascii"),
            "finite": bool(torch.isfinite(vector).all().item()),
            "abs_max": float(torch.nan_to_num(vector).abs().max().item()),
            "tp_rank": 0,
            "tp_size": _tp_size(),
        }
        _EMITTED[boundary] += 1
        print(_MARKER + json.dumps(event, separators=(",", ":")), flush=True)


def trace_sglang_attention_detail(
    layer_id: int,
    suffix: str,
    tensor: torch.Tensor,
    forward_batch,
) -> None:
    """Trace one TP-local attention boundary when both backends use equal TP."""

    if not should_trace_attention_detail(layer_id):
        return
    trace_sglang_prefill_hidden(
        f"layer.{layer_id}.attn.{suffix}",
        tensor,
        forward_batch.input_ids,
        forward_batch,
    )
