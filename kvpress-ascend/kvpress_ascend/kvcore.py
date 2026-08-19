# SPDX-FileCopyrightText: Copyright (c) 2025 kvpress-ascend contributors
# SPDX-License-Identifier: Apache-2.0
"""Core KV-cache compression mechanics for the vLLM v1 worker.

Layout used by the engine
-------------------------
A request owns ``m = ceil(orig_len / bs)`` blocks after prefill.  Compression
keeps ``n_kept`` tokens per layer with ``delta = orig_len - n_kept``.  The
physical cache is rewritten *in place within the request's own tail blocks*:
kept (key, value) pairs of every layer are written densely into the last
``k = m - delta // bs`` blocks ``[b_{m-k} .. b_{m-1}]`` of the request's block
table.  The front blocks keep their original content (so prefix-cache hits on
the front part remain valid), and the evicted middle region is never
referenced again.

Choosing ``k = m - delta // bs`` (instead of ``ceil(n_kept / bs)``) gives the
kept region at least as much slot slack as the original prompt had
(``k*bs - n_kept >= m*bs - orig_len``), which guarantees that the compressed
positions never outrun the scheduler's block-table growth: a new token at
compressed position ``n_kept + j`` maps to block index ``k + floor((j-d')/bs)``
whose physical block ``b_{m + floor((j-d')/bs)}`` the scheduler has already
allocated for its own position ``orig_len + j``.

Every subsequent step the engine rewrites the *logical* view of the request:

* block-table row: ``row' = [b_{m-k} .. b_{m-1}] + [b_m ..]``
* slot mapping:    positions of the request are shifted by ``delta`` so that
                   new tokens land at slot ``n_kept + j``,
* seq lens:        ``seq_len' = seq_len - delta``.

All functions in this module are device agnostic and operate on plain torch
tensors / numpy arrays, so they are exercised by the offline simulation.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

import numpy as np

from .log import get_logger
from .registry import CompressionRecord

logger = get_logger()

_LAYER_IDX_RE = re.compile(r"layers\.(\d+)")


def extract_layer_index(layer_name: str) -> int:
    m = _LAYER_IDX_RE.search(layer_name)
    return int(m.group(1)) if m else 0


# --------------------------------------------------------------------------- #
# Cache tensor access
# --------------------------------------------------------------------------- #


def split_kv_cache(kv_cache) -> Tuple[object, object]:
    """Return (key_cache, value_cache) from an ascend/upstream kv_cache.

    vllm-ascend binds ``layer.kv_cache`` to either a tensor of shape
    (2, num_blocks, bs, kv_heads, head_dim) or a (key, value) sequence.
    """
    if kv_cache is None:
        return None, None
    if isinstance(kv_cache, (list, tuple)):
        if len(kv_cache) >= 2:
            return kv_cache[0], kv_cache[1]
        if len(kv_cache) == 1 and isinstance(kv_cache[0], (list, tuple)):
            return kv_cache[0][0], kv_cache[0][1]
        return None, None
    if hasattr(kv_cache, "shape") and len(kv_cache.shape) > 0 and kv_cache.shape[0] == 2:
        return kv_cache[0], kv_cache[1]
    return None, None


def cache_views(key_cache, value_cache, num_kv_heads: int, head_dim: int):
    """Return flat views of the paged caches for slot indexing.

    Ascend cache layout: (num_blocks, block_size, num_kv_heads, head_size).
    Falls back to a reshape if the leading dim is 2 (upstream layout
    (2, num_blocks, ...) kept together).
    """
    import torch

    def _flat(cache):
        if cache is None:
            return None
        if isinstance(cache, torch.Tensor):
            if cache.dim() == 5 and cache.shape[0] == 2:
                cache = cache[0]
            return cache.view(-1, num_kv_heads, head_dim)
        return cache

    return _flat(key_cache), _flat(value_cache)


def slot_ids_for_positions(row: np.ndarray, positions: np.ndarray, block_size: int) -> np.ndarray:
    """Physical cache slot ids for absolute token positions in a request.

    slot = row[position // bs] * bs + position % bs
    """
    positions = np.asarray(positions, dtype=np.int64)
    block_ids = row[np.asarray(positions // block_size, dtype=np.int64)]
    return (block_ids * block_size + positions % block_size).astype(np.int64)


def _flatten_cache(cache, num_kv_heads: int, head_dim: int):
    """Reshape a paged cache tensor to (num_blocks*bs, kv_heads, head_dim)."""
    import torch

    if cache is None:
        return None
    if isinstance(cache, torch.Tensor):
        if cache.dim() == 5 and cache.shape[0] == 2:
            cache = cache[0]
        return cache.reshape(-1, num_kv_heads, head_dim)
    return cache


def gather_request_kv(
    key_cache,
    value_cache,
    row: np.ndarray,
    orig_len: int,
    block_size: int,
    num_kv_heads: int,
    head_dim: int,
    device,
    dtype,
) -> Tuple[object, object]:
    """Gather a request's dense (1, kv_heads, seq, head_dim) K/V from cache."""
    import torch

    key_flat = _flatten_cache(key_cache, num_kv_heads, head_dim)
    value_flat = _flatten_cache(value_cache, num_kv_heads, head_dim)
    positions = np.arange(orig_len, dtype=np.int64)
    slots = slot_ids_for_positions(row, positions, block_size)
    slots_t = torch.from_numpy(slots).to(device)
    keys = torch.index_select(key_flat, 0, slots_t).transpose(0, 1).unsqueeze(0).to(dtype)
    values = torch.index_select(value_flat, 0, slots_t).transpose(0, 1).unsqueeze(0).to(dtype)
    return keys, values


def compact_layer_inplace(
    key_flat,
    value_flat,
    row: np.ndarray,
    record: CompressionRecord,
    keep_idx: np.ndarray,
    num_kv_heads: int,
    head_dim: int,
    device,
    dtype,
    dry_run: bool = False,
    n_kept_override: Optional[int] = None,
) -> None:
    """Write the kept K/V of one layer into the request's tail blocks.

    keep_idx: (num_kv_heads, n_kept) positions (prompt-relative) to keep.
    The kept vectors land at slots [0, n_kept) of the *compressed* row, i.e.
    into physical blocks [b_{m-k} .. b_{m-1}] with block offsets 0..bs-1.
    ``n_kept_override`` allows a per-layer keep count (SqueezeAttention).
    """
    import torch

    m = record.n_blocks_orig
    n_kept = record.n_kept if n_kept_override is None else int(n_kept_override)
    if n_kept <= 0:
        return
    delta = record.orig_len - n_kept
    k = max(m - (delta // record.block_size), 1)
    bs = record.block_size

    if keep_idx is None:
        return

    key_flat = _flatten_cache(key_flat, num_kv_heads, head_dim)
    value_flat = _flatten_cache(value_flat, num_kv_heads, head_dim)

    # destination: dense slots of the tail region [b_{m-k} .. b_{m-1}]
    tail_blocks = row[m - k : m]
    slots = np.concatenate([tail_blocks[i] * bs + np.arange(bs, dtype=np.int64) for i in range(k)])[:n_kept]
    slots_t = torch.from_numpy(slots).to(device)

    keep_t = torch.from_numpy(keep_idx).to(device)  # (kv_heads, n_kept)

    # source: original slots of the whole prompt
    orig_slots = slot_ids_for_positions(row, np.arange(record.orig_len, dtype=np.int64), bs)
    orig_slots_t = torch.from_numpy(orig_slots).to(device)

    flat_k = key_flat.view(-1, num_kv_heads, head_dim)
    flat_v = value_flat.view(-1, num_kv_heads, head_dim)
    src_k = torch.index_select(flat_k, 0, orig_slots_t)  # (orig_len, kv_heads, hd)
    src_v = torch.index_select(flat_v, 0, orig_slots_t)

    def _per_head(src):
        # (orig_len, kv_heads, hd) -> (kv_heads, n_kept, hd)
        src = src.permute(1, 0, 2)
        return torch.gather(src, 1, keep_t.unsqueeze(-1).expand(-1, -1, head_dim))

    kept_k = _per_head(src_k)
    kept_v = _per_head(src_v)

    if dry_run:
        return

    # in-place write through the flat views (they alias the paged cache)
    flat_k.index_copy_(0, slots_t, kept_k.permute(1, 0, 2).contiguous())
    flat_v.index_copy_(0, slots_t, kept_v.permute(1, 0, 2).contiguous())


# --------------------------------------------------------------------------- #
# Block-table row rewrite
# --------------------------------------------------------------------------- #


def rewrite_block_table_row(
    row_np: np.ndarray,
    row_width: int,
    cur_blocks: int,
    record: CompressionRecord,
) -> None:
    """Rewrite one request's row in place (shared/uniform layout).

    Compressed row::

        row' = [b_{m-k} .. b_{m-1}] + [b_m .. b_{cur_blocks-1}]

    i.e. the k tail blocks that now hold the compacted KV, followed by the
    scheduler-grown blocks.  ``num_blocks_per_row`` bookkeeping is
    intentionally untouched: only the block-id *content* changes.
    """
    m = record.n_blocks_orig
    k = record.n_blocks_kept
    if k >= m:
        # Nothing to compress (kept region spans the whole prompt).
        return
    kept = row_np[m - k : m].copy()          # tail blocks holding kept KV
    grown = row_np[m:cur_blocks].copy()      # scheduler-grown blocks
    row_np[:] = 0
    row_np[:k] = kept
    row_np[k : k + grown.size] = grown


def per_layer_row(record: CompressionRecord, row_np: np.ndarray, cur_blocks: int, n_kept_layer: int) -> np.ndarray:
    """Build a per-layer compressed row (tail-based layout).

    row'_l = [b_{m-k_l} .. b_{m-1}] + [b_m .. b_{cur_blocks-1}]
    with k_l = m - floor((orig_len - n_kept_layer) / bs).
    """
    m = record.n_blocks_orig
    delta_l = record.orig_len - n_kept_layer
    k_l = max(m - (delta_l // record.block_size), 1)
    out = np.zeros_like(row_np)
    if k_l > 0:
        out[:k_l] = row_np[m - k_l : m]
    grown = row_np[m:cur_blocks]
    if grown.size > 0:
        out[k_l : k_l + grown.size] = grown
    return out


# --------------------------------------------------------------------------- #
# Seq-lens / positions
# --------------------------------------------------------------------------- #


def corrected_seq_len(record: CompressionRecord, seq_len: int) -> int:
    return seq_len - record.delta


def shifted_positions_tensor(positions, query_start_loc, req_order: List[str], registry, device) -> object:
    """Return a copy of ``positions`` with compressed requests shifted by delta.

    ``positions``: (num_tokens,) int64 tensor in TND order.
    ``query_start_loc``: (num_reqs + 1,) cumulative token counts (GPU tensor).
    ``req_order``: batch row order == input_batch.req_ids.

    All arithmetic stays on the device: per-request deltas are built on CPU
    (no device sync) and gathered per token with repeat_interleave.
    """
    import torch

    num_reqs = len(req_order)
    if num_reqs == 0 or positions.numel() == 0:
        return positions
    deltas = np.zeros(num_reqs, dtype=np.int64)
    any_compressed = False
    for i, req_id in enumerate(req_order):
        record = registry.get(req_id)
        if record is not None:
            deltas[i] = record.delta
            any_compressed = True
    if not any_compressed:
        return positions
    deltas_t = torch.from_numpy(deltas).to(device)
    qsl = query_start_loc.to(device) if query_start_loc.device != device else query_start_loc
    req_idx = torch.repeat_interleave(torch.arange(num_reqs, device=device), qsl[1:] - qsl[:-1])
    if req_idx.numel() > positions.numel():
        req_idx = req_idx[: positions.numel()]
    out = positions.clone()
    out -= deltas_t[req_idx.long()]
    return out


def shifted_positions_numpy(
    req_indices: np.ndarray,
    positions: np.ndarray,
    req_order: List[str],
    registry,
) -> np.ndarray:
    """numpy variant used by ``compute_slot_mapping_draft``."""
    out = np.asarray(positions, dtype=np.int64).copy()
    req_indices = np.asarray(req_indices, dtype=np.int64)
    for i, req_id in enumerate(req_order):
        record = registry.get(req_id)
        if record is None:
            continue
        mask = req_indices == i
        if np.any(mask):
            out[mask] = out[mask] - record.delta
    return out


# --------------------------------------------------------------------------- #
# Keep-index selection
# --------------------------------------------------------------------------- #


def select_keep_indices(scores, n_kept: int, device) -> np.ndarray:
    """Top-k per kv-head; returns (kv_heads, n_kept) int64 CPU array."""
    import torch

    # scores: (1, kv_heads, seq_len)
    flat = scores[0]
    if n_kept <= 0:
        n_kept = 1
    n_kept = min(n_kept, flat.shape[-1])
    _, indices = torch.topk(flat, n_kept, dim=-1)
    return indices.detach().to("cpu").numpy().astype(np.int64)


def compute_scores(
    press,
    layer_idx: int,
    keys,
    values,
    queries,
    attentions,
    pooled_scores: Optional[Dict[int, object]],
    extra: Optional[dict] = None,
) -> object:
    """Run the press score callable, handling cross-layer pooling."""
    scores = press.score(layer_idx, keys, values, queries, attentions, extra)
    if press.pool_across_layers:
        if pooled_scores is not None:
            pooled_scores[layer_idx] = scores
        return None
    return scores


def finalize_pooled_scores(press, pooled_scores: Dict[int, object], device) -> object:
    """Average per-layer scores into one (1, kv_heads, seq_len) tensor."""
    import torch

    if not pooled_scores:
        return None
    items = list(pooled_scores.values())
    if not items:
        return None
    out = items[0]
    for s in items[1:]:
        out = out + s
    return out / len(items)
