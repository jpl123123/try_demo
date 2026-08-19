# SPDX-FileCopyrightText: Copyright (c) 2025 kvpress-ascend contributors
# SPDX-License-Identifier: Apache-2.0
"""Runtime state: compression records, per-step capture context, registry.

The engine keeps two kinds of state:

* ``CaptureContext`` - set for the duration of one model-runner step.  It maps
  the TND batch row order of the attention backends (which equals
  ``input_batch.req_ids`` order) to request ids, tells which requests are
  still prefilling, and remembers the prefill-start token counts (used to
  detect requests that hit the prefix cache).

* ``CompressionRecord`` - created when a request's prefill completes and the
  request is actually compressed.  It stores the original prompt length, the
  number of kept tokens, the per-layer keep indices (CPU) and the geometry of
  the compressed layout (block size, tail-block offset, delta).

Records are consulted on every subsequent step by:

* the input-batch block-table row rewrite (``kvcore.rewrite_block_table_row``),
* the slot-mapping position shift (patched ``BlockTable.compute_slot_mapping``),
* the attention-metadata seq-lens correction
  (patched ``NPUModelRunner._build_attention_metadata``).
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


@dataclass
class CompressionRecord:
    req_id: str
    orig_len: int            # prompt length at compression time
    n_kept: int              # tokens kept per layer (uniform layout)
    block_size: int          # physical block size (ascend: 128)
    keep_indices: Dict[int, np.ndarray] = field(default_factory=dict)  # layer -> (kv_heads, n_kept)
    layer_names: List[str] = field(default_factory=list)
    layer_n_kept: Dict[int, int] = field(default_factory=dict)  # per-layer keep counts (squeeze)

    @property
    def delta(self) -> int:
        return self.orig_len - self.n_kept

    @property
    def n_blocks_orig(self) -> int:
        return int(np.ceil(self.orig_len / self.block_size))

    @property
    def n_blocks_kept(self) -> int:
        """Number of blocks of the compressed row (kept region).

        k = m - floor(delta / bs).  This is >= ceil(n_kept / bs) and guarantees
        that the block-space advance rate of the compressed positions never
        outruns the scheduler's own block table growth (see kvcore module
        docstring).  Clamped to at least 1 block.
        """
        m = self.n_blocks_orig
        k = m - (self.delta // self.block_size)
        return max(int(k), 1)


@dataclass
class CaptureContext:
    """Per-step context bridging the model runner and the attention layers."""

    step_id: int
    # batch order == input_batch.req_ids order (TND order of the backends)
    req_ids: List[str]
    # per request: number of tokens scheduled this step (target model)
    num_scheduled_tokens: np.ndarray
    # per request: number of computed tokens BEFORE this step
    num_computed_tokens_before: np.ndarray
    # per request: prompt length
    num_prompt_tokens: np.ndarray
    # True when the step is a prefill step at all
    is_prefill_step: bool = False
    # per-request capture buffers (ReqCapture objects keyed by req_id)
    captures: dict = field(default_factory=dict)
    # per-layer captured query tail (1, num_heads, window, head_dim) of the
    # final prefill chunk; only filled for presses with needs_queries
    queries: Dict[int, object] = field(default_factory=dict)
    # per-layer captured hidden-state pairs for squeeze-style importance
    hidden_cos_sim: Dict[int, float] = field(default_factory=dict)

    def req_index(self, req_id: str) -> int:
        try:
            return self.req_ids.index(req_id)
        except ValueError:
            return -1

    def is_prefilling(self, req_id: str) -> bool:
        idx = self.req_index(req_id)
        if idx < 0:
            return False
        return bool(self.num_computed_tokens_before[idx] < self.num_prompt_tokens[idx])

    def prefill_hit_prefix_cache(self, req_id: str) -> bool:
        idx = self.req_index(req_id)
        if idx < 0:
            return False
        # A request whose prefill started with computed tokens > 0 reused
        # prefix-cache blocks.  Compression rewrites cache content, which is
        # unsafe for shared blocks.
        return bool(self.num_computed_tokens_before[idx] > 0)


class Registry:
    """Process-wide registry of compression records and the active press."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.records: Dict[str, CompressionRecord] = {}
        self.press = None
        self.extra_capture_hooks = []  # callables(layer_idx, **tensors) -> dict
        self.step_counter = 0
        self.active_context: Optional[CaptureContext] = None
        self.stats = {
            "requests_seen": 0,
            "prefills_completed": 0,
            "compressed": 0,
            "skipped_short": 0,
            "skipped_prefix_cache": 0,
            "skipped_shared_row": 0,
            "skipped_error": 0,
            "dry_run": 0,
        }

    # ---- records ---------------------------------------------------------- #
    def put(self, record: CompressionRecord) -> None:
        with self._lock:
            self.records[record.req_id] = record

    def get(self, req_id: str) -> Optional[CompressionRecord]:
        with self._lock:
            return self.records.get(req_id)

    def drop(self, req_id: str) -> None:
        with self._lock:
            self.records.pop(req_id, None)

    def __contains__(self, req_id: str) -> bool:
        with self._lock:
            return req_id in self.records

    # ---- step context ----------------------------------------------------- #
    def begin_step(self, ctx: CaptureContext) -> None:
        self.step_counter += 1
        self.active_context = ctx

    def end_step(self) -> None:
        self.active_context = None

    def current(self) -> Optional[CaptureContext]:
        return self.active_context

    def bump(self, key: str, delta: int = 1) -> None:
        with self._lock:
            self.stats[key] = self.stats.get(key, 0) + delta


REGISTRY = Registry()
