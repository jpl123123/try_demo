# SPDX-FileCopyrightText: Copyright (c) 2025 kvpress-ascend contributors
# SPDX-License-Identifier: Apache-2.0
"""Offline simulation harness: fake vllm/vllm_ascend objects + step driver.

vllm / vllm-ascend / torch-npu cannot run on this development machine, so the
engine is exercised against faithful mocks of the exact surfaces it patches.
The mocks replicate the vllm 0.23.0 / vllm-ascend 0.23.0 contracts verified
from the vllm-ascend sources:

* ``NPUModelRunner`` with ``input_batch`` (req_ids, req_id_to_index,
  num_computed_tokens_cpu, num_prompt_tokens, block_table), ``kv_cache_config``
  (kv_cache_groups[0] = FullAttentionSpec), ``compilation_config``
  (static_forward_context), ``cache_config``, ``positions``, ``query_start_loc``
* ``BlockTable`` with ``block_table`` (CpuGpuBuffer .np/.gpu),
  ``num_blocks_per_row``, ``commit_block_table`` and the slot-mapping kernel
  semantics
* ``AscendMetadata`` (seq_lens, seq_lens_cpu, seq_lens_list, block_tables,
  slot_mapping, actual_seq_lengths_q, attn_state)

The driver simulates the vllm v1 step order used by the patches:

    execute_model_pre -> (scheduler fills block rows) -> _prepare_inputs
    (row rewrite) -> compute_slot_mapping (position shift) -> backend forward
    (query capture) -> attention module forward (hidden capture) -> KV write
    -> execute_model_post (compression pass) -> num_computed update
"""

from __future__ import annotations

import os
import sys
import types
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

# --------------------------------------------------------------------------- #
# Fake objects
# --------------------------------------------------------------------------- #


class CpuGpuBuffer:
    def __init__(self, np_arr: np.ndarray):
        self.np = np_arr
        self.cpu = np_arr

    def fill_(self, v):
        self.np.fill(v)

    @property
    def gpu(self):
        return self

    def __getitem__(self, item):
        return self.np[item]

    def __setitem__(self, key, value):
        self.np[key] = value


class FakeBlockTable:
    """Mirrors vllm_ascend.worker.block_table.BlockTable for group 0."""

    def __init__(self, block_size: int, max_num_blocks_per_req: int, max_num_reqs: int):
        self.block_size = block_size
        self.max_num_blocks_per_req = max_num_blocks_per_req
        self.block_table = CpuGpuBuffer(np.zeros((max_num_reqs, max_num_blocks_per_req), dtype=np.int32))
        self.num_blocks_per_row = np.zeros(max_num_reqs, dtype=np.int32)
        self.slot_mapping = CpuGpuBuffer(np.zeros(max_num_reqs * 4096, dtype=np.int32))
        self._fake_gpu_tensor = None

    def append_row(self, block_ids, row_idx):
        block_ids = np.asarray(block_ids)
        start = int(self.num_blocks_per_row[row_idx])
        self.block_table.np[row_idx, start : start + len(block_ids)] = block_ids
        self.num_blocks_per_row[row_idx] += len(block_ids)

    def commit_block_table(self, num_reqs):
        pass

    def compute_slot_mapping(self, num_reqs, query_start_loc, positions):
        """Kernel semantics: slot = row[position // bs] * bs + position % bs."""
        positions = positions.numpy() if hasattr(positions, "numpy") else np.asarray(positions)
        qsl = query_start_loc.numpy() if hasattr(query_start_loc, "numpy") else np.asarray(query_start_loc)
        slots = np.zeros(positions.shape[0], dtype=np.int64)
        for i in range(num_reqs):
            start, end = int(qsl[i]), int(qsl[i + 1])
            p = positions[start:end]
            row = self.block_table.np[i]
            block_ids = row[np.asarray(p // self.block_size, dtype=np.int64)]
            slots[start:end] = block_ids * self.block_size + p % self.block_size
        self.slot_mapping.np[: slots.shape[0]] = slots
        return slots

    def compute_slot_mapping_draft(self, req_indices, positions):
        req_indices = np.asarray(req_indices)
        positions = np.asarray(positions)
        slots = np.zeros(positions.shape[0], dtype=np.int64)
        for i in range(len(req_indices)):
            row = self.block_table.np[req_indices[i]]
            block_ids = row[positions[i] // self.block_size]
            slots[i] = block_ids * self.block_size + positions[i] % self.block_size
        self.slot_mapping.np[: slots.shape[0]] = slots
        return slots


@dataclass
class FakeInputBatch:
    req_ids: List[str] = field(default_factory=list)
    req_id_to_index: Dict[str, int] = field(default_factory=dict)
    num_computed_tokens_cpu: np.ndarray = None
    num_prompt_tokens: np.ndarray = None
    num_computed_tokens: np.ndarray = None
    block_table: Optional[FakeBlockTable] = None


@dataclass
class FakeKVCacheSpec:
    block_size: int = 128
    num_kv_heads: int = 8
    head_size: int = 128
    dtype: str = "auto"


class FakeFullAttentionSpec(FakeKVCacheSpec):
    pass


@dataclass
class FakeKVCacheGroup:
    layer_names: List[str] = field(default_factory=list)
    kv_cache_spec: FakeKVCacheSpec = field(default_factory=FakeFullAttentionSpec)


@dataclass
class FakeKVCacheConfig:
    kv_cache_groups: List[FakeKVCacheGroup] = field(default_factory=list)


@dataclass
class FakeCompilationConfig:
    static_forward_context: Dict[str, object] = field(default_factory=dict)


@dataclass
class FakeCacheConfig:
    enable_prefix_caching: bool = False


class FakeLayerModule:
    """Stands in for an Attention layer; holds the bound KV cache."""

    def __init__(self, kv_cache):
        self.kv_cache = kv_cache


@dataclass
class FakeSchedulerOutput:
    total_num_scheduled_tokens: int = 0
    num_scheduled_tokens: np.ndarray = None
    scheduled_cached_reqs: object = None
    scheduled_new_reqs: list = field(default_factory=list)
    finished_reqs: list = field(default_factory=list)


@dataclass
class FakeScheduledReqs:
    req_ids: List[str] = field(default_factory=list)
    num_computed_tokens: np.ndarray = None


@dataclass
class FakeAttnMeta:
    """Stands in for AscendMetadata."""

    seq_lens: object = None
    seq_lens_cpu: object = None
    seq_lens_list: list = None
    block_tables: object = None
    slot_mapping: object = None
    actual_seq_lengths_q: list = None
    attn_state: str = "ChunkedPrefill"
    num_actual_tokens: int = 0
    num_prefills: int = 1
    num_decodes: int = 0


@dataclass
class FakeCommonAttnMeta:
    seq_lens: object = None
    seq_lens_cpu: object = None
    _seq_lens_cpu: object = None
    block_table_tensor: object = None
    slot_mapping: object = None
    query_start_loc: object = None
    num_reqs: int = 0
    num_actual_tokens: int = 0


class FakeRunner:
    """Stands in for NPUModelRunner."""

    def __init__(
        self,
        layer_names: List[str],
        block_size: int = 128,
        num_kv_heads: int = 8,
        head_size: int = 128,
        max_num_blocks_per_req: int = 64,
        max_num_reqs: int = 16,
        prefix_caching: bool = False,
    ):
        self.block_size = block_size
        self.kv_cache_config = FakeKVCacheConfig(
            kv_cache_groups=[
                FakeKVCacheGroup(layer_names=list(layer_names), kv_cache_spec=FakeFullAttentionSpec(
                    block_size=block_size, num_kv_heads=num_kv_heads, head_size=head_size
                ))
            ]
        )
        self.compilation_config = FakeCompilationConfig()
        self.cache_config = FakeCacheConfig(enable_prefix_caching=prefix_caching)
        self.input_batch = FakeInputBatch(block_table=FakeBlockTable(block_size, max_num_blocks_per_req, max_num_reqs))
        self.positions = CpuGpuBuffer(np.zeros(max_num_reqs * 4096, dtype=np.int64))
        self.query_start_loc = CpuGpuBuffer(np.zeros(max_num_reqs + 1, dtype=np.int32))
        self._max_num_reqs = max_num_reqs

    def register_layer(self, layer_name: str, kv_cache):
        self.compilation_config.static_forward_context[layer_name] = FakeLayerModule(kv_cache)

    def set_requests(self, req_ids: List[str], num_computed: np.ndarray, num_prompt: np.ndarray):
        self.input_batch.req_ids = list(req_ids)
        self.input_batch.req_id_to_index = {r: i for i, r in enumerate(req_ids)}
        n = self._max_num_reqs
        self.input_batch.num_computed_tokens_cpu = np.zeros(n, dtype=np.int64)
        self.input_batch.num_computed_tokens = np.zeros(n, dtype=np.int64)
        self.input_batch.num_prompt_tokens = np.zeros(n, dtype=np.int64)
        for i, r in enumerate(req_ids):
            self.input_batch.num_computed_tokens_cpu[i] = num_computed[i]
            self.input_batch.num_computed_tokens[i] = num_computed[i]
            self.input_batch.num_prompt_tokens[i] = num_prompt[i]

    # methods wrapped by the engine (no-ops in the simulation; the driver
    # calls the engine hooks directly in the right order)
    def execute_model(self, scheduler_output, intermediate_tensors=None):
        return None

    def _prepare_inputs(self, scheduler_output, num_scheduled_tokens):
        return None, None, 0

    def _build_attention_metadata(self, *args, **kwargs):
        return {}, None


# --------------------------------------------------------------------------- #
# Fake module tree (installed into sys.modules for engine.apply)
# --------------------------------------------------------------------------- #


def _make_module(name: str):
    mod = types.ModuleType(name)
    sys.modules[name] = mod
    return mod


def install_fake_vllm() -> None:
    """Install the fake vllm/vllm_ascend modules required by engine.apply."""
    for name in ("vllm", "vllm.model_executor", "vllm.model_executor.layers", "vllm.model_executor.layers.attention"):
        _make_module(name)
    for name in (
        "vllm_ascend",
        "vllm_ascend.attention",
        "vllm_ascend.attention.attention_v1",
        "vllm_ascend.worker",
        "vllm_ascend.worker.block_table",
        "vllm_ascend.worker.model_runner_v1",
        "vllm_ascend.ascend_forward_context",
    ):
        _make_module(name)

    vllm_attn = sys.modules["vllm.model_executor.layers.attention"]
    extra_ctx = sys.modules["vllm_ascend.ascend_forward_context"]
    attn_v1 = sys.modules["vllm_ascend.attention.attention_v1"]
    bt_mod = sys.modules["vllm_ascend.worker.block_table"]
    mr_mod = sys.modules["vllm_ascend.worker.model_runner_v1"]

    class _EXTRA_CTX:
        is_draft_model = False

    extra_ctx._EXTRA_CTX = _EXTRA_CTX

    class FakeAttention:
        forward = None  # patched by engine

    vllm_attn.Attention = FakeAttention

    class FakeAttentionBackendImpl:
        forward = None

    class FakeC8AttentionBackendImpl(FakeAttentionBackendImpl):
        forward = None

    attn_v1.AscendAttentionBackendImpl = FakeAttentionBackendImpl
    attn_v1.AscendC8AttentionBackendImpl = FakeC8AttentionBackendImpl
    bt_mod.BlockTable = FakeBlockTable
    mr_mod.NPUModelRunner = FakeRunner


def uninstall_fake_vllm() -> None:
    for name in list(sys.modules):
        if name == "vllm" or name.startswith("vllm.") or name.startswith("vllm_ascend"):
            sys.modules.pop(name, None)


# --------------------------------------------------------------------------- #
# Step driver
# --------------------------------------------------------------------------- #


class CacheSim:
    """Paged KV cache + synthetic model data for one simulated request."""

    def __init__(self, num_blocks: int, block_size: int, num_kv_heads: int, head_size: int, dtype="float32"):
        import torch

        self.torch = torch
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.num_kv_heads = num_kv_heads
        self.head_size = head_size
        self.key_cache = torch.randn(num_blocks, block_size, num_kv_heads, head_size, dtype=getattr(torch, dtype))
        self.value_cache = torch.randn(num_blocks, block_size, num_kv_heads, head_size, dtype=getattr(torch, dtype))

    def write(self, slots: np.ndarray, keys, values):
        """keys/values: (T, kv_heads, head_dim) -- vllm reshape_and_cache."""
        import torch

        kf = self.key_cache.reshape(-1, self.num_kv_heads, self.head_size)
        vf = self.value_cache.reshape(-1, self.num_kv_heads, self.head_size)
        slots_t = torch.from_numpy(slots.astype(np.int64))
        kf.index_copy_(0, slots_t, keys)
        vf.index_copy_(0, slots_t, values)

    def read(self, slots: np.ndarray):
        import torch

        kf = self.key_cache.reshape(-1, self.num_kv_heads, self.head_size)
        vf = self.value_cache.reshape(-1, self.num_kv_heads, self.head_size)
        slots_t = torch.from_numpy(slots.astype(np.int64))
        return torch.index_select(kf, 0, slots_t), torch.index_select(vf, 0, slots_t)


class FakeScheduler:
    """Grows block-table rows like the vllm v1 scheduler."""

    def __init__(self, block_size: int):
        self.block_size = block_size
        self.next_block = 0

    def alloc_blocks(self, n: int) -> list:
        blocks = list(range(self.next_block, self.next_block + n))
        self.next_block += n
        return blocks

    def grow_rows(self, runner: FakeRunner, req_index: int, num_tokens: int):
        bt = runner.input_batch.block_table
        needed = int(np.ceil(num_tokens / self.block_size))
        have = int(bt.num_blocks_per_row[req_index])
        if needed > have:
            bt.append_row(self.alloc_blocks(needed - have), req_index)


def simulate_step(
    runner: FakeRunner,
    engine,
    scheduler: FakeScheduler,
    caches: Dict[int, CacheSim],
    req_ids: List[str],
    num_scheduled: np.ndarray,
    chunk_kv: Dict[int, object],
    chunk_queries: Dict[int, object],
    chunk_hidden: Dict[int, object],
    layer_names: List[str],
    chunk_attn_meta: FakeAttnMeta,
    chunk_hidden_out: Optional[Dict[int, object]] = None,
) -> None:
    """One simulated vllm v1 step (prefill or decode).

    chunk_kv / chunk_queries / chunk_hidden: per-layer tensors of the tokens
    scheduled this step (T, kv_heads, hd) / (T, heads, hd) / (T, H).
    caches: per-layer CacheSim instances.
    chunk_hidden_out: optional per-layer attention outputs (defaults to the
    input, which yields cos-sim 1.0).
    """
    import torch

    num_reqs = len(req_ids)
    total = int(num_scheduled.sum())
    sched = FakeSchedulerOutput(
        total_num_scheduled_tokens=total,
        num_scheduled_tokens=num_scheduled,
        scheduled_cached_reqs=FakeScheduledReqs(req_ids=req_ids, num_computed_tokens=num_scheduled),
    )
    engine.on_execute_model_pre(runner, sched)

    # scheduler grows rows
    for i, r in enumerate(req_ids):
        idx = runner.input_batch.req_id_to_index[r]
        after = int(runner.input_batch.num_computed_tokens_cpu[idx]) + int(num_scheduled[i])
        scheduler.grow_rows(runner, idx, after)

    # _prepare_inputs entry: row rewrite + positions/query_start_loc
    engine.on_prepare_inputs_entry(runner)
    positions = torch.from_numpy(np.concatenate(
        [np.arange(int(runner.input_batch.num_computed_tokens_cpu[runner.input_batch.req_id_to_index[r]]),
                  int(runner.input_batch.num_computed_tokens_cpu[runner.input_batch.req_id_to_index[r]]) + int(num_scheduled[i]))
         for i, r in enumerate(req_ids)]
    )).long()
    qsl = np.zeros(num_reqs + 1, dtype=np.int64)
    qsl[1:] = np.cumsum(num_scheduled)
    runner.query_start_loc.np[: num_reqs + 1] = qsl
    runner.positions = positions.clone()
    shifted = engine.on_compute_slot_mapping(num_reqs, torch.from_numpy(qsl), positions)
    slots = runner.input_batch.block_table.compute_slot_mapping(num_reqs, torch.from_numpy(qsl), shifted)

    # backend forward (query capture) + attention module forward (hidden)
    for layer_name in layer_names:
        layer_idx = int(layer_name.split(".")[2])
        meta = chunk_attn_meta
        meta.actual_seq_lengths_q = [int(x) for x in num_scheduled]
        fake_layer = types.SimpleNamespace(layer_name=layer_name)
        engine.on_backend_forward(fake_layer, chunk_queries.get(layer_idx), meta, is_draft=False)
        h_out = chunk_hidden_out.get(layer_idx) if chunk_hidden_out else None
        engine.on_attention_module_forward(
            fake_layer, chunk_hidden.get(layer_idx), meta, h_out if h_out is not None else chunk_hidden.get(layer_idx)
        )

    # KV write (reshape_and_cache semantics) into each layer's own cache
    for layer_name in layer_names:
        layer_idx = int(layer_name.split(".")[2])
        kv = chunk_kv.get(layer_idx)
        if kv is not None and layer_idx in caches:
            caches[layer_idx].write(slots, kv[0], kv[1])

    # execute_model_post: compression pass
    engine.on_execute_model_post(runner, sched, None)

    # update bookkeeping
    for i, r in enumerate(req_ids):
        idx = runner.input_batch.req_id_to_index[r]
        runner.input_batch.num_computed_tokens_cpu[idx] += num_scheduled[i]
        runner.input_batch.num_computed_tokens[idx] += num_scheduled[i]


# --------------------------------------------------------------------------- #
# CLI self-check
# --------------------------------------------------------------------------- #


def run_self_check() -> int:
    """Offline end-to-end self check (no NPU / vllm needed)."""
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from kvpress_ascend import presses, registry as R
    from kvpress_ascend.engine import Engine

    torch = __import__("torch")
    BS_ = 128
    KV = 2
    HEADS = 4
    HD = 16
    HID = 64
    LAYERS_ = ["model.layers.0.self_attn.attn", "model.layers.1.self_attn.attn"]

    def rchunk(n, seed):
        torch.manual_seed(seed)
        kv, q, h = {}, {}, {}
        for ln in LAYERS_:
            li = int(ln.split(".")[2])
            kv[li] = (torch.randn(n, KV, HD), torch.randn(n, KV, HD))
            q[li] = torch.randn(n, HEADS, HD)
            h[li] = torch.randn(n, HID)
        return kv, q, h

    engine = Engine(registry=R.Registry())
    press = presses.build_press("snapkv", 0.5, window=64)
    runner = FakeRunner(layer_names=LAYERS_, block_size=BS_, num_kv_heads=KV, head_size=HD, max_num_blocks_per_req=64)
    caches = {}
    for ln in LAYERS_:
        li = int(ln.split(".")[2])
        caches[li] = CacheSim(num_blocks=512, block_size=BS_, num_kv_heads=KV, head_size=HD)
        runner.register_layer(ln, (caches[li].key_cache, caches[li].value_cache))
    scheduler = FakeScheduler(BS_)
    engine.press = press
    engine.min_len = 0
    runner.set_requests(["req0"], np.array([0], dtype=np.int64), np.array([300], dtype=np.int64))
    kv1, q1, h1 = rchunk(128, 1)
    kv2, q2, h2 = rchunk(172, 2)
    meta = FakeAttnMeta()
    simulate_step(runner, engine, scheduler, caches, ["req0"], np.array([128]), kv1, q1, h1, LAYERS_, meta)
    simulate_step(runner, engine, scheduler, caches, ["req0"], np.array([172]), kv2, q2, h2, LAYERS_, meta)
    rec = engine.registry.get("req0")
    if rec is None or rec.n_kept != 150:
        print("FAIL: request was not compressed")
        return 1
    print("compression record ok: n_kept =", rec.n_kept, "| stats:", dict(engine.registry.stats))
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(run_self_check())
