# SPDX-FileCopyrightText: Copyright (c) 2025 kvpress-ascend contributors
# SPDX-License-Identifier: Apache-2.0
"""Self-contained KV-cache compression policies (kvpress-style scorers).

The original ``kvpress`` library scores dense (bsz, kv_heads, seq, head_dim)
keys during a transformers prefill pass.  In vLLM the same scoring data is
reconstructed per layer from the paged KV cache plus, for attention-based
scorers, the tail of the per-layer queries captured during the final prefill
chunk.  All scorers here operate on the shapes vLLM can provide:

* ``keys`` / ``values`` : (1, num_kv_heads, seq_len, head_dim)
* ``queries``          : (1, num_heads, window, head_dim)  post-RoPE tail
* ``attentions``       : (1, num_heads, window, seq_len)   causal window
                         attention (computed by the engine when queries are
                         available and the scorer needs them)

Scores are always (1, num_kv_heads, seq_len); higher = keep.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

# torch is imported lazily (module must stay importable without torch at
# interpreter startup when the patch is disabled).
TORCH = None


def _torch():
    global TORCH
    if TORCH is None:
        import torch

        TORCH = torch
    return TORCH


# --------------------------------------------------------------------------- #
# Press interface
# --------------------------------------------------------------------------- #


@dataclass
class Press:
    """A compression policy understood by the kvpress-ascend engine.

    ``score`` receives per-layer tensors and returns scores
    (1, kv_heads, seq_len).  ``needs_queries`` asks the engine to capture the
    post-RoPE query tail of the final prefill chunk; ``pool_across_layers``
    makes the engine average this layer's scores with all other layers before
    selecting the keep set (PyramidKV-style).
    """

    name: str
    compression_ratio: float = 0.5
    needs_queries: bool = False
    needs_hidden: bool = False
    pool_across_layers: bool = False
    extra: dict = field(default_factory=dict)

    # score(layer_idx, keys, values, queries, attentions, extra=None)
    #   -> (1, kv_heads, L)
    score: Callable = field(default=None, repr=False)  # type: ignore[assignment]

    def __post_init__(self):
        if self.score is None:
            raise ValueError(f"Press {self.name} has no score callable")
        if not (0 <= self.compression_ratio < 1):
            raise ValueError("compression_ratio must be in [0, 1)")

    # -- optional per-layer budget hooks (used by SqueezeAttention) --------- #

    def prepare(self, layer_names: list, orig_len: int, extra: dict) -> None:
        """Called once per request before scoring; may compute budgets."""

    def resolved_n_kept(self, orig_len: int, req_id: str = None):
        """Override the global n_kept; None = use compression_ratio."""
        return None

    def n_kept_layer(self, layer_idx: int, orig_len: int, req_id: str = None):
        """Per-layer keep count; None = global n_kept."""
        return None


def _window_attention(queries, keys, window: int, num_kv_heads: int, head_dim: int):
    """Causal attention of the last `window` queries over all keys.

    Mirrors kvpress SnapKVPress.compute_window_attention: softmax over the
    full causal row, caller slices later.
    queries: (1, num_heads, window, head_dim) post-RoPE
    keys:    (1, num_kv_heads, seq_len, head_dim)
    Returns (1, num_heads, window, seq_len) fp32 weights.
    """
    torch = _torch()
    bsz, num_heads, w, _ = queries.shape
    _, num_kv_heads, k_len, _ = keys.shape
    groups = num_heads // num_kv_heads
    if groups > 1:
        keys = keys.repeat_interleave(groups, dim=1)
    attn = torch.matmul(queries, keys.transpose(2, 3)) / math.sqrt(head_dim)
    mask = torch.full_like(attn, float("-inf"))
    mask = torch.triu(mask, diagonal=k_len - w + 1)
    attn = attn + mask
    attn = torch.softmax(attn, dim=-1, dtype=torch.float32)
    return attn


# --------------------------------------------------------------------------- #
# Scorers
# --------------------------------------------------------------------------- #


def make_random(ratio: float, seed: Optional[int] = None) -> Press:
    torch = _torch()

    def score(layer_idx, keys, values, queries, attentions, extra=None):
        generator = None
        if seed is not None:
            generator = torch.Generator(device=keys.device)
            generator.manual_seed(seed)
        return torch.rand(*keys.shape[:-1], generator=generator, device=keys.device, dtype=keys.dtype)

    return Press(name="random", compression_ratio=ratio, score=score)


def make_streamingllm(ratio: float, n_sink: int = 4) -> Press:
    torch = _torch()

    def score(layer_idx, keys, values, queries, attentions, extra=None):
        k_len = keys.shape[2]
        if k_len <= n_sink:
            return torch.ones_like(keys[..., 0])
        n_pruned = k_len - int(k_len * (1 - ratio))
        scores = torch.ones_like(keys[..., 0])
        scores[:, :, n_sink : n_sink + n_pruned] = 0
        return scores

    press = Press(name="streamingllm", compression_ratio=ratio, score=score)
    press.n_sink = n_sink  # core parameter, visible to the heartbeat
    return press


def make_knorm(ratio: float) -> Press:
    torch = _torch()

    def score(layer_idx, keys, values, queries, attentions, extra=None):
        return -keys.norm(dim=-1)

    return Press(name="knorm", compression_ratio=ratio, score=score)


def make_snapkv(ratio: float, window: int = 64, kernel_size: int = 5) -> Press:
    torch = _torch()

    def score(layer_idx, keys, values, queries, attentions, extra=None):
        bsz, num_kv_heads, k_len, _ = keys.shape
        num_heads = queries.shape[1]
        groups = num_heads // num_kv_heads
        w = min(window, k_len - 1)
        if queries is None or queries.shape[2] < w:
            # Not enough captured queries (or capture disabled): fall back to
            # a positional score so the pipeline still runs.
            return torch.ones_like(keys[..., 0])
        q = queries[:, :, -w:, :]
        attn = _window_attention(q, keys, w, num_kv_heads, keys.shape[-1])
        attn_weights = attn[..., : k_len - w]
        scores = attn_weights.mean(dim=-2)
        if kernel_size > 1:
            scores = torch.nn.functional.avg_pool1d(
                scores, kernel_size=kernel_size, padding=kernel_size // 2, stride=1
            )
        scores = scores.view(bsz, num_kv_heads, groups, k_len - w).mean(2)
        scores = torch.nn.functional.pad(scores, (0, w), value=float(scores.max().item() + 1))
        return scores

    press = Press(name="snapkv", compression_ratio=ratio, needs_queries=True, score=score)
    press.window = window  # core parameter, visible to the heartbeat
    press.kernel_size = kernel_size
    return press


def make_tova(ratio: float) -> Press:
    torch = _torch()

    def score(layer_idx, keys, values, queries, attentions, extra=None):
        bsz, num_kv_heads, k_len, _ = keys.shape
        w = 1
        if queries is None or queries.shape[2] < 1:
            return torch.ones_like(keys[..., 0])
        q = queries[:, :, -1:, :]
        attn = _window_attention(q, keys, w, num_kv_heads, keys.shape[-1])
        attn_weights = attn[..., : k_len - 1]
        scores = attn_weights.mean(1)
        scores = scores.repeat(1, num_kv_heads, 1)
        scores = torch.nn.functional.pad(scores, (0, 1), value=float(scores.max().item() + 1))
        return scores

    return Press(name="tova", compression_ratio=ratio, needs_queries=True, score=score)


def make_pyramidkv(ratio: float, window: int = 64) -> Press:
    base = make_snapkv(ratio, window=window)
    base.name = "pyramidkv"
    base.pool_across_layers = True
    return base


def make_adakv(ratio: float, alpha_safeguard: float = 0.2, base: Optional[Press] = None) -> Press:
    """Head-wise top-k with a per-head safeguard (AdaKV-style).

    The uniform block layout forces every head to keep exactly n_kept slots;
    AdaKV's cross-head global bottom-k pruning is approximated by a per-head
    top-k where the top alpha*n_kept tokens of every head are protected.
    """
    torch = _torch()
    if base is None:
        base = make_snapkv(ratio)

    def score(layer_idx, keys, values, queries, attentions, extra=None):
        s = base.score(layer_idx, keys, values, queries, attentions)
        k_len = s.shape[-1]
        n_kept = int(k_len * (1 - ratio))
        n_safe = max(int(n_kept * alpha_safeguard), 1)
        top = torch.topk(s, n_safe, dim=-1).indices
        s.scatter_(-1, top, torch.finfo(s.dtype).max)
        return s

    press = Press(name="adakv", compression_ratio=ratio, needs_queries=base.needs_queries, score=score)
    press.alpha_safeguard = alpha_safeguard  # core parameter, visible to the heartbeat
    return press


_REGISTRY = {
    "random": make_random,
    "streamingllm": make_streamingllm,
    "knorm": make_knorm,
    "snapkv": make_snapkv,
    "tova": make_tova,
    "pyramidkv": make_pyramidkv,
    "adakv": make_adakv,
}


def build_press(name: str, ratio: float, window: int = 64, sink: int = 4) -> Press:
    name = name.strip().lower()
    if name not in _REGISTRY:
        raise ValueError(
            f"Unknown KVPRESS_ASCEND_PRESS={name!r}. "
            f"Available: {sorted(_REGISTRY)}"
        )
    if name == "streamingllm":
        return make_streamingllm(ratio, n_sink=sink)
    if name == "snapkv":
        return make_snapkv(ratio, window=window)
    if name == "tova":
        return make_tova(ratio)
    if name == "pyramidkv":
        return make_pyramidkv(ratio, window=window)
    if name == "adakv":
        return make_adakv(ratio, base=make_snapkv(ratio, window=window))
    return _REGISTRY[name](ratio)


# --------------------------------------------------------------------------- #
# KMeans helper used by the SqueezeAttention press (kept here so both packages
# share one implementation; sklearn is used when available).
# --------------------------------------------------------------------------- #


def kmeans_1d(values: np.ndarray, n_clusters: int, n_init: int = 10, seed: int = 0) -> np.ndarray:
    """Cluster a 1-D array into n_clusters groups; return cluster labels.

    Falls back to a deterministic quantile split if sklearn is unavailable.
    """
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    try:
        from sklearn.cluster import KMeans

        km = KMeans(n_clusters=n_clusters, n_init=n_init, random_state=seed)
        labels = km.fit_predict(values.reshape(-1, 1))
        # Rename labels so that label 0 == smallest cluster centre.
        centers = km.cluster_centers_.flatten()
        order = np.argsort(centers)
        remap = np.empty(n_clusters, dtype=int)
        remap[order] = np.arange(n_clusters)
        return remap[labels]
    except Exception:  # pragma: no cover - fallback path
        qs = np.quantile(values, np.linspace(0, 1, n_clusters + 1)[1:-1])
        labels = np.searchsorted(qs, values, side="right")
        return labels.astype(int)
