# SPDX-FileCopyrightText: Copyright (c) 2025 SqueezeAttention-ascend contributors
# SPDX-License-Identifier: Apache-2.0
"""Offline simulation tests for SqueezeAttention-ascend.

Run with::

    python -m pytest tests/ -v
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "kvpress-ascend"))

from kvpress_ascend import registry as registry_mod  # noqa: E402
from kvpress_ascend.engine import Engine  # noqa: E402
from kvpress_ascend.simulate import (  # noqa: E402
    CacheSim,
    FakeAttnMeta,
    FakeRunner,
    FakeScheduler,
    simulate_step,
)

from squeezeattention_ascend.press import SqueezePress  # noqa: E402

BS = 128
KV_HEADS = 2
HEADS = 4
HEAD_DIM = 16
HIDDEN = 64
# 4 layers so the 3-cluster allocation has a real split
LAYERS = [
    "model.layers.0.self_attn.attn",
    "model.layers.1.self_attn.attn",
    "model.layers.2.self_attn.attn",
    "model.layers.3.self_attn.attn",
]


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for key in ("SQUEEZE", "SQUEEZE_ASCEND", "KVPRESS", "KVPRESS_ASCEND"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("SQUEEZE", "1")


def _make_runner(prefix_caching=False, speculative=False):
    runner = FakeRunner(
        layer_names=LAYERS,
        block_size=BS,
        num_kv_heads=KV_HEADS,
        head_size=HEAD_DIM,
        max_num_blocks_per_req=64,
        max_num_reqs=16,
        prefix_caching=prefix_caching,
    )
    runner.speculative_config = {"method": "qwen3_5_mtp"} if speculative else None
    return runner


def _register_caches(runner, num_blocks=1024):
    caches = {}
    for layer_name in LAYERS:
        layer_idx = int(layer_name.split(".")[2])
        caches[layer_idx] = CacheSim(num_blocks=num_blocks, block_size=BS, num_kv_heads=KV_HEADS, head_size=HEAD_DIM)
        runner.register_layer(layer_name, (caches[layer_idx].key_cache, caches[layer_idx].value_cache))
    return caches


def _rchunk(n, seed=0):
    torch.manual_seed(seed)
    kv, q, h = {}, {}, {}
    for ln in LAYERS:
        li = int(ln.split(".")[2])
        kv[li] = (torch.randn(n, KV_HEADS, HEAD_DIM), torch.randn(n, KV_HEADS, HEAD_DIM))
        q[li] = torch.randn(n, HEADS, HEAD_DIM)
        h[li] = torch.randn(n, HIDDEN)
    return kv, q, h


def _hidden_out(h, n):
    """Layer-dependent attention outputs so per-layer cos-sim differs."""
    torch.manual_seed(7)
    out = {}
    for li, hin in h.items():
        noise = torch.randn_like(hin) * (0.15 + 0.15 * li)
        out[li] = hin + noise
    return out


def _run_prefill(engine, press, speculative=False, prefix_caching=False, orig_len=300, per_layer_mode=True):
    runner = _make_runner(prefix_caching=prefix_caching, speculative=speculative)
    caches = _register_caches(runner)
    scheduler = FakeScheduler(BS)
    engine.set_press(press, per_layer_mode=per_layer_mode, capture_hidden=True)
    engine.min_len = 0
    engine.registry.records.clear()
    engine.registry.stats.clear()
    runner.set_requests(["req0"], np.array([0], dtype=np.int64), np.array([orig_len], dtype=np.int64))
    kv, q, h = _rchunk(orig_len, seed=3)
    hout = _hidden_out(h, orig_len)
    meta = FakeAttnMeta()
    simulate_step(runner, engine, scheduler, caches, ["req0"], np.array([orig_len]), kv, q, h, LAYERS, meta, hout)
    return runner, caches, scheduler


def test_budget_allocation_preserves_total():
    press = SqueezePress(ini_size=0.21, class3_ratio=0.08, n_sink=4)
    engine = Engine(registry=registry_mod.Registry())
    runner, caches, scheduler = _run_prefill(engine, press)
    record = engine.registry.get("req0")
    assert record is not None
    # total budget must be preserved: sum(n_kept_l) == num_layers * ini_size * orig_len
    budgets = record.layer_n_kept
    assert len(budgets) == len(LAYERS)
    total = sum(budgets.values())
    target = len(LAYERS) * 0.21 * 300
    assert abs(total - target) <= len(LAYERS) * 300 * 0.05, f"total budget {total} vs target {target}"
    # per-layer budgets must differ (the point of SqueezeAttention)
    assert len(set(budgets.values())) > 1


def test_per_layer_layout_end_to_end():
    press = SqueezePress(ini_size=0.21, class3_ratio=0.08, n_sink=4)
    engine = Engine(registry=registry_mod.Registry())
    engine.per_layer_mode = True
    runner, caches, scheduler = _run_prefill(engine, press)
    record = engine.registry.get("req0")
    assert record is not None
    assert len(record.layer_n_kept) == len(LAYERS)

    # verify per-layer keep sets: sink + recent per layer budget
    for li in record.layer_n_kept:
        n_kept_l = record.layer_n_kept[li]
        keep = np.sort(record.keep_indices[li][0])
        expected = np.concatenate([np.arange(4), np.arange(300 - (n_kept_l - 4), 300)])
        assert np.array_equal(keep, expected), f"layer {li}: keep set mismatch"

    # decode step: per-layer metadata must be corrected with per-layer lens
    bt = runner.input_batch.block_table
    row = bt.block_table.np[0].copy()
    kv3, q3, h3 = _rchunk(1, seed=9)
    meta = FakeAttnMeta()
    simulate_step(runner, engine, scheduler, caches, ["req0"], np.array([1]), kv3, q3, h3, LAYERS, meta)
    # rebuild metadata the way the engine sees it and correct it
    from kvpress_ascend.kvcore import per_layer_row

    engine._per_layer_rows["req0"] = {
        li: per_layer_row(record, row, int(bt.num_blocks_per_row[0]), record.layer_n_kept[li])
        for li in record.layer_n_kept
    }
    metas = {}
    for ln in LAYERS:
        li = int(ln.split(".")[2])
        metas[ln] = FakeAttnMeta(
            seq_lens=torch.tensor([301], dtype=torch.int64),
            seq_lens_cpu=torch.tensor([301], dtype=torch.int64),
            seq_lens_list=[301],
            block_tables=torch.from_numpy(np.stack([row] * 16)).clone(),
            slot_mapping=torch.zeros(1, dtype=torch.int64),
            actual_seq_lengths_q=[1],
        )
    runner.positions = torch.tensor([300], dtype=torch.int64)
    runner.query_start_loc.np[:2] = [0, 1]
    engine.on_build_attention_metadata(runner, metas, None)
    for ln in LAYERS:
        li = int(ln.split(".")[2])
        assert int(metas[ln].seq_lens[0]) == record.layer_n_kept[li] + 1, f"layer {li} seq-lens correction"
    # per-layer block tables must differ where budgets differ
    rows = [metas[ln].block_tables[0].tolist() for ln in LAYERS]
    assert any(r != rows[0] for r in rows[1:]), "per-layer block tables should differ"


def test_uniform_layout_under_spec_decode():
    """With MTP active the engine must fall back to the uniform layout."""
    press = SqueezePress(ini_size=0.21, class3_ratio=0.08, n_sink=4)
    engine = Engine(registry=registry_mod.Registry())
    engine.per_layer_mode = True  # requested, but spec decode disables it
    runner, caches, scheduler = _run_prefill(engine, press, speculative=True)
    record = engine.registry.get("req0")
    assert record is not None
    # all layers share the global budget
    assert len(set(record.layer_n_kept.values())) <= 1
    assert engine._use_per_layer_layout(runner) is False


def test_env_gating(monkeypatch):
    import squeezeattention_ascend.envs as envs

    monkeypatch.delenv("SQUEEZE", raising=False)
    monkeypatch.delenv("squeeze", raising=False)
    monkeypatch.delenv("SQUEEZE_ASCEND", raising=False)
    assert envs.enabled() is False
    monkeypatch.setenv("squeeze", "1")
    assert envs.enabled() is True
    assert envs.ini_size() == 0.21
    monkeypatch.setenv("SQUEEZE_ASCEND_INI_SIZE", "0.3")
    assert envs.ini_size() == 0.3


def test_kmeans_helper():
    from kvpress_ascend.presses import kmeans_1d

    values = np.array([0.1, 0.12, 0.5, 0.55, 0.9, 0.95])
    labels = kmeans_1d(values, 3, seed=0)
    assert labels.shape == (6,)
    # cluster ids ordered by centre
    centres = [values[labels == c].mean() for c in range(3)]
    assert centres == sorted(centres)
