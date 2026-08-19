# SPDX-FileCopyrightText: Copyright (c) 2025 SqueezeAttention-ascend contributors
# SPDX-License-Identifier: Apache-2.0
"""Lightweight logger for SqueezeAttention-ascend."""

from __future__ import annotations

import logging
import sys

_LEVELS = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
}


def get_logger() -> logging.Logger:
    logger = logging.getLogger("squeezeattention_ascend")
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter("[squeeze-ascend] %(levelname)s %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


def configure(level: str) -> None:
    get_logger().setLevel(_LEVELS.get(level.lower(), logging.INFO))
