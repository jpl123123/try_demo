# SPDX-FileCopyrightText: Copyright (c) 2025 SqueezeAttention-ascend contributors
# SPDX-License-Identifier: Apache-2.0
"""SqueezeAttention-ascend: layer-wise KV-cache budget allocation for vllm-ascend.

Activation: the installed ``squeezeattention_ascend.pth`` imports this module
at interpreter startup, but everything stays inert unless an activation
variable is set, e.g.::

    export squeeze=1

The policy runs on the shared kvpress-ascend patch engine (installed as a
dependency): the engine captures per-layer hidden-state similarities during
prefill, this package allocates the per-layer budgets with the SqueezeAttention
KMeans scheme, and the engine compacts the paged KV cache accordingly.
"""

from __future__ import annotations

from . import envs
from .log import configure, get_logger

logger = get_logger()

__all__ = ["enabled", "apply", "envs"]


def enabled() -> bool:
    return envs.enabled()


def apply() -> None:
    """Register the squeeze policy on the kvpress-ascend engine."""
    try:
        from kvpress_ascend import engine as kp_engine

        engine = kp_engine.get_engine()
        from .press import register

        register(engine)
        # make sure the engine patches are installed (idempotent)
        kp_engine.apply()
    except Exception as exc:  # pragma: no cover - defensive
        logger.error("SqueezeAttention-ascend failed to activate: %s", exc)


def _auto_apply() -> None:
    if not envs.enabled():
        return
    configure(envs.log_level())
    logger.info("SqueezeAttention-ascend activated via environment")
    apply()


_auto_apply()
