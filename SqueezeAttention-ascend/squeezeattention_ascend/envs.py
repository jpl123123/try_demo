# SPDX-FileCopyrightText: Copyright (c) 2025 SqueezeAttention-ascend contributors
# SPDX-License-Identifier: Apache-2.0
"""Environment configuration for SqueezeAttention-ascend.

Activation (any truthy value)::

    export squeeze=1
    export SQUEEZE=1
    export SQUEEZE_ASCEND=1
    export SQUEEZE_ASCEND_ENABLED=1

Knobs::

    SQUEEZE_ASCEND_INI_SIZE       average budget per layer as a fraction of
                                  the prompt length      (default 0.21)
    SQUEEZE_ASCEND_CLASS3_RATIO   budget fraction of the most important layer
                                  class                  (default 0.08)
    SQUEEZE_ASCEND_SINK           streaming-LLM sink tokens (default 4)
    SQUEEZE_ASCEND_KMEANS_SEED    KMeans seed (default 0)
    SQUEEZE_ASCEND_PER_LAYER      per-layer layouts (default 1; automatically
                                  disabled under speculative decoding, where
                                  the uniform layout is required)

Requires ``kvpress-ascend`` to be installed (the shared patch engine).
"""

from __future__ import annotations

import os

_TRUTHY = {"1", "true", "on", "yes", "y", "1.0"}

_ACTIVATION_KEYS = (
    "SQUEEZE_ASCEND_ENABLED",
    "SQUEEZE_ASCEND",
    "SQUEEZEATTENTION",
    "SQUEEZE",
    "squeeze",
)


def enabled() -> bool:
    for key in _ACTIVATION_KEYS:
        val = os.environ.get(key)
        if val is not None and val.strip().lower() in _TRUTHY:
            return True
    return False


def ini_size() -> float:
    return float(os.environ.get("SQUEEZE_ASCEND_INI_SIZE", "0.21"))


def class3_ratio() -> float:
    return float(os.environ.get("SQUEEZE_ASCEND_CLASS3_RATIO", "0.08"))


def sink() -> int:
    return int(os.environ.get("SQUEEZE_ASCEND_SINK", "4"))


def kmeans_seed() -> int:
    return int(os.environ.get("SQUEEZE_ASCEND_KMEANS_SEED", "0"))


def per_layer_layout() -> bool:
    return os.environ.get("SQUEEZE_ASCEND_PER_LAYER", "1").strip().lower() in _TRUTHY


def log_level() -> str:
    return os.environ.get("SQUEEZE_ASCEND_LOG", "info").strip().lower()
