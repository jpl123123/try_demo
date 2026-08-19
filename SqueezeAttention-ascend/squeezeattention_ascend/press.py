# SPDX-FileCopyrightText: Copyright (c) 2025 SqueezeAttention-ascend contributors
# SPDX-License-Identifier: Apache-2.0
"""SqueezeAttention policy for the kvpress-ascend engine.

SqueezeAttention (Wang & Gan, 2024, arXiv:2404.04793) allocates the KV-cache
budget across layers *on the fly*: during prefill it measures each decoder
layer's importance as the cosine similarity between the layer's input and
output hidden states, clusters the layers with KMeans into three classes and
assigns each class a budget (``ini_size`` of the prompt length on average,
``class3_ratio`` for the most important class).  Each layer then keeps the
budgeted number of KV pairs in a streaming-LLM fashion (sink tokens + most
recent tokens).

This policy plugs into the kvpress-ascend engine:

* ``needs_hidden=True`` activates the engine's per-layer hidden-state capture
  (input/output of every attention module during prefill chunks),
* ``prepare()`` runs the KMeans allocation once per request,
* per-layer budgets are applied through ``n_kept_layer()``.  When speculative
  decoding (MTP/EAGLE) is active the engine falls back to a uniform layout
  (mean budget) because the draft model shares the group-0 cache with a single
  block layout.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from kvpress_ascend.presses import Press, kmeans_1d

from . import envs


class SqueezePress(Press):
    def __init__(
        self,
        ini_size: float = 0.21,
        class3_ratio: float = 0.08,
        n_sink: int = 4,
        n_clusters: int = 3,
        seed: int = 0,
    ) -> None:
        super().__init__(name="squeeze", compression_ratio=0.0, needs_hidden=True, score=self._score)
        self.ini_size = ini_size
        self.class3_ratio = class3_ratio
        self.n_sink = n_sink
        self.n_clusters = n_clusters
        self.seed = seed
        self._budgets: Dict[str, Dict[int, int]] = {}   # req_id -> layer -> n_kept
        self._global_n_kept: Dict[str, int] = {}        # req_id -> mean budget
        self._per_layer: Dict[str, bool] = {}           # req_id -> layout mode

    # ------------------------------------------------------------------ #

    def _layer_importance(self, extra: dict) -> Optional[Dict[int, float]]:
        """Mean cosine similarity per layer from the engine's captures."""
        ctx = extra.get("ctx")
        req_id = extra.get("req_id")
        if ctx is None or req_id is None:
            return None
        cap = ctx.captures.get(req_id)
        if cap is None or not cap.cos_sims:
            return None
        out = {}
        for layer_idx, values in cap.cos_sims.items():
            if values:
                out[layer_idx] = float(np.mean(values))
        return out or None

    def prepare(self, layer_names: list, orig_len: int, extra: dict) -> None:
        req_id = extra.get("req_id")
        if req_id is None:
            return
        per_layer = bool(extra.get("per_layer", True))
        self._per_layer[req_id] = per_layer

        n_layers = len(layer_names)
        importance = self._layer_importance(extra)

        if importance is None or len(importance) < 3 or n_layers < 3:
            # capture failed (e.g. signature mismatch): uniform budget
            budget = max(int(orig_len * self.ini_size), self.n_sink + 1)
            self._budgets[req_id] = {int(i.split(".")[2]): budget for i in layer_names}
            self._global_n_kept[req_id] = budget
            return

        means = np.array([importance[int(i.split(".")[2])] for i in layer_names], dtype=np.float64)
        labels = kmeans_1d(means, self.n_clusters, seed=self.seed)
        centers = np.array(
            [means[labels == c].mean() if np.any(labels == c) else -np.inf for c in range(self.n_clusters)]
        )
        class3 = int(np.argmax(centers))
        n_class3 = int((labels == class3).sum())
        n_other = n_layers - n_class3
        if n_other <= 0:
            a = self.ini_size
        else:
            a = (n_layers * self.ini_size - n_class3 * self.class3_ratio) / n_other

        budgets: Dict[int, int] = {}
        for i, layer_name in enumerate(layer_names):
            layer_idx = int(layer_name.split(".")[2])
            ratio = self.class3_ratio if labels[i] == class3 else a
            n = int(orig_len * ratio)
            n = max(min(n, orig_len - 1), self.n_sink + 1)
            budgets[layer_idx] = n

        self._budgets[req_id] = budgets
        self._global_n_kept[req_id] = int(round(float(np.mean(list(budgets.values())))))
        self._log_allocation(req_id, layer_names, labels, class3, a, budgets)

    def _log_allocation(self, req_id, layer_names, labels, class3, a, budgets) -> None:
        from .log import get_logger

        logger = get_logger()
        per_layer = {
            int(n.split(".")[2]): int(labels[i])
            for i, n in enumerate(layer_names)
        }
        logger.info(
            "req %s: SqueezeAttention budgets (class3=%s a=%.4f): %s",
            req_id,
            class3,
            a,
            {k: v for k, v in sorted(budgets.items())},
        )
        logger.debug("req %s: layer classes: %s", req_id, per_layer)

    def resolved_n_kept(self, orig_len: int, req_id: str = None):
        if req_id is not None and req_id in self._global_n_kept:
            return self._global_n_kept[req_id]
        return int(orig_len * self.ini_size)

    def n_kept_layer(self, layer_idx: int, orig_len: int, req_id: str = None):
        if req_id is not None and self._per_layer.get(req_id, False):
            budgets = self._budgets.get(req_id)
            if budgets and layer_idx in budgets:
                return budgets[layer_idx]
        return None  # uniform layout: engine uses the global n_kept

    # ------------------------------------------------------------------ #

    def _score(self, layer_idx, keys, values, queries, attentions, extra=None):
        """Streaming-LLM style positional score with the layer's own budget."""
        import torch

        req_id = (extra or {}).get("req_id")
        per_layer = self._per_layer.get(req_id, False) if req_id else False
        budgets = self._budgets.get(req_id) if req_id else None
        if per_layer and budgets and layer_idx in budgets:
            n_kept_l = budgets[layer_idx]
        elif req_id is not None:
            n_kept_l = self._global_n_kept.get(req_id)
        else:
            n_kept_l = None
        if n_kept_l is None:
            n_kept_l = int(keys.shape[2] * self.ini_size)
        k_len = keys.shape[2]
        n_sink = min(self.n_sink, k_len - 1)
        n_kept_l = max(min(int(n_kept_l), k_len - 1), n_sink + 1)
        recent = max(n_kept_l - n_sink, 1)
        scores = torch.zeros_like(keys[..., 0])
        scores[:, :, :n_sink] = 1.0
        scores[:, :, -recent:] = 1.0
        return scores


def build_squeeze_press() -> SqueezePress:
    return SqueezePress(
        ini_size=envs.ini_size(),
        class3_ratio=envs.class3_ratio(),
        n_sink=envs.sink(),
        seed=envs.kmeans_seed(),
    )


def register(engine) -> SqueezePress:
    """Register the squeeze policy on the kvpress-ascend engine."""
    from .log import get_logger

    logger = get_logger()
    press = build_squeeze_press()
    try:
        from kvpress_ascend import envs as kp_envs

        force = kp_envs.policy() == "squeeze"
    except Exception:
        force = False
    engine.set_press(
        press,
        per_layer_mode=envs.per_layer_layout(),
        capture_hidden=True,
        force=force,
    )
    logger.info(
        "SqueezeAttention-ascend registered: ini_size=%.3f class3_ratio=%.3f sink=%d",
        press.ini_size,
        press.class3_ratio,
        press.n_sink,
    )
    return press
