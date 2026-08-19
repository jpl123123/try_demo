# SPDX-FileCopyrightText: Copyright (c) 2025 kvpress-ascend contributors
# SPDX-License-Identifier: Apache-2.0
"""kvpress-ascend: monkeypatch adapter porting NVIDIA kvpress to vllm-ascend.

Activation
----------
This module is imported automatically at interpreter startup through the
installed ``kvpress_ascend.pth`` file, but it is completely inert unless one
of the activation environment variables is truthy, e.g.::

    export kvpress=1

or::

    export KVPRESS=1

See ``kvpress_ascend.envs`` for the full list of knobs.
"""

from __future__ import annotations

from . import envs
from .log import get_logger

logger = get_logger()

__all__ = ["apply", "enabled", "envs"]


def enabled() -> bool:
    return envs.enabled()


def apply() -> None:
    """Install the monkeypatches into the running vllm-ascend process."""
    from . import engine

    engine.apply()


def _auto_apply() -> None:
    if not envs.enabled():
        return
    from .log import configure

    configure(envs.log_level())
    logger.info("kvpress-ascend activated via environment (policy=%s)", envs.policy())
    try:
        apply()
    except Exception as exc:  # pragma: no cover - defensive
        logger.error("kvpress-ascend failed to activate: %s", exc)


_auto_apply()
