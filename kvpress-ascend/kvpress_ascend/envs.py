# SPDX-FileCopyrightText: Copyright (c) 2025 kvpress-ascend contributors
# SPDX-License-Identifier: Apache-2.0
"""Environment-variable configuration for kvpress-ascend.

Activation
----------
The package installs a ``kvpress_ascend.pth`` file into site-packages so that
``kvpress_ascend`` is imported at interpreter startup (this covers ``vllm
serve``, its engine-core subprocesses and its TP workers).  The patch is
*inert* unless one of the activation variables below is set to a truthy value
(``1``, ``true``, ``on``, ``yes``).  Any of the following enables the patch::

    export kvpress=1            # exactly as requested: "export kvpress"
    export KVPRESS=1
    export KVPRESS_ASCEND=1
    export KVPRESS_ASCEND_ENABLED=1

Alternatively (or in addition) you may load the package explicitly through
vLLM's own plugin mechanism::

    export VLLM_PLUGINS=kvpress_ascend

Behaviour knobs (all optional)
------------------------------
KVPRESS_ASCEND_PRESS      compression policy: snapkv | streamingllm | random |
                          knorm | tova | pyramidkv | adakv   (default snapkv)
KVPRESS_ASCEND_RATIO      compression ratio in [0,1)         (default 0.5)
KVPRESS_ASCEND_WINDOW     SnapKV/TOVA observation window     (default 64)
KVPRESS_ASCEND_SINK       streaming-llm sink tokens          (default 4)
KVPRESS_ASCEND_MIN_LEN    do not compress prompts shorter than this (default 2048)
KVPRESS_ASCEND_PREFIX_CACHE  skip | force                    (default skip)
                          With vLLM prefix caching enabled the compressed tail
                          blocks would keep stale hashes in the engine-core
                          block pool; 'skip' refuses to compress (safe), 'force'
                          compresses anyway (may corrupt future prefix-cache
                          hits -- use only when you accept that).
KVPRESS_ASCEND_DRY_RUN   1 -> compute scores, log statistics, but do not
                          rewrite the KV cache (default 0)
KVPRESS_ASCEND_LOG       debug | info | warning               (default info)
KVPRESS_ASCEND_POLICY    kvpress | squeeze                    (default kvpress)
                          Only relevant when SqueezeAttention-ascend is also
                          enabled (both .pth files active).
"""

from __future__ import annotations

import os

_TRUTHY = {"1", "true", "on", "yes", "y", "1.0"}

_ACTIVATION_KEYS = (
    "KVPRESS_ASCEND_ENABLED",
    "KVPRESS_ASCEND",
    "KVPRESS",
    "kvpress",
)

_POLICY_KEYS = ("KVPRESS_ASCEND_POLICY",)


def _env(*names: str, default: str | None = None) -> str | None:
    for name in names:
        val = os.environ.get(name)
        if val is not None and val.strip() != "":
            return val.strip()
    return default


def enabled() -> bool:
    for key in _ACTIVATION_KEYS:
        val = os.environ.get(key)
        if val is not None and val.strip().lower() in _TRUTHY:
            return True
    return False


def policy() -> str:
    val = _env(*_POLICY_KEYS, default="kvpress")
    return val.strip().lower()


def press() -> str:
    return _env("KVPRESS_ASCEND_PRESS", default="snapkv").strip().lower()


def ratio() -> float:
    return float(_env("KVPRESS_ASCEND_RATIO", default="0.5"))


def window() -> int:
    return int(_env("KVPRESS_ASCEND_WINDOW", default="64"))


def sink() -> int:
    return int(_env("KVPRESS_ASCEND_SINK", default="4"))


def min_len() -> int:
    return int(_env("KVPRESS_ASCEND_MIN_LEN", default="2048"))


def prefix_cache_mode() -> str:
    return _env("KVPRESS_ASCEND_PREFIX_CACHE", default="skip").strip().lower()


def dry_run() -> bool:
    return _env("KVPRESS_ASCEND_DRY_RUN", default="0").strip().lower() in _TRUTHY


def log_level() -> str:
    return _env("KVPRESS_ASCEND_LOG", default="info").strip().lower()


def step_log() -> bool:
    """Per-inference heartbeat: log at every step whether both patches are in
    their core code paths (seam probes) and with which core parameters."""
    return _env("KVPRESS_ASCEND_STEP_LOG", default="1").strip().lower() in _TRUTHY


def spec_decode_allow() -> bool:
    # Speculative decoding (MTP/EAGLE) is fully supported by the uniform-layout
    # engine.  This knob is kept for future per-layer layouts.
    return _env("KVPRESS_ASCEND_SPEC_DECODE", default="allow").strip().lower() in _TRUTHY


def summarize() -> dict:
    return {
        "enabled": enabled(),
        "policy": policy(),
        "press": press(),
        "ratio": ratio(),
        "window": window(),
        "sink": sink(),
        "min_len": min_len(),
        "prefix_cache_mode": prefix_cache_mode(),
        "dry_run": dry_run(),
        "log_level": log_level(),
    }
