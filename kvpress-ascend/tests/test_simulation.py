# SPDX-FileCopyrightText: Copyright (c) 2025 kvpress-ascend contributors
# SPDX-License-Identifier: Apache-2.0
"""Offline simulation tests.

These tests run the kvpress-ascend engine against fake vllm/vllm-ascend
objects on CPU (no NPU, no vllm required).  They exercise the exact code
paths that run inside a vllm-ascend worker, including the compression pass,
the block-table row rewrite, the slot-mapping position shift and the
attention-metadata seq-lens correction, and they validate the end-to-end
invariant: after compression, attention over the compressed layout equals
attention over the reference kept tokens.

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

from kvpress_ascend import engine as engine_mod  # noqa: E402
from kvpress_ascend import envs, presses, registry as registry_mod  # noqa: E402
from kvpress_ascend.engine import Engine  # noqa: E402
from kvpress_ascend.presses import Press  # noqa: E402
from kvpress_ascend.simulate import (  # noqa: E402
    CacheSim,
    FakeAttnMeta,
    FakeRunner,
    FakeScheduler,
    install_fake_vllm,
    simulate_step,
    uninstall_fake_vllm,
)

BS = 128
KV_HEADS = 2
HEADS = 4
HEAD_DIM = 16
HIDDEN = 64
LAYERS = ["model.layers.0.self_attn.attn", "model.layers.1.self_attn.attn"]


def _make_runner(prefix_caching: bool = False) -> FakeRunner:
    return FakeRunner(
        layer_names=LAYERS,
        block_size=BS,
        num_kv_heads=KV_HEADS,
        head_size=HEAD_DIM,
        max_num_blocks_per_req=64,
        max_num_reqs=16,
        prefix_caching=prefix_caching,
    )


def _make_caches(num_blocks: int = 512) -> Dict[int, CacheSim]:
    caches = {}
    for layer_name in LAYERS:
        layer_idx = int(layer_name.split(".")[2])
        caches[layer_idx] = CacheSim(num_blocks=num_blocks, block_size=BS, num_kv_heads=KV_HEADS, head_size=HEAD_DIM)
    return caches


def _register_caches(runner: FakeRunner, caches: Dict[int, CacheSim]):
    for layer_name in LAYERS:
        layer_idx = int(layer_name.split(".")[2])
        c = caches[layer_idx]
        runner.register_layer(layer_name, (c.key_cache, c.value_cache))


def _random_chunk(rng: np.random.Generator, n_tokens: int):
    """Random per-layer K/V/query/hidden data for one chunk."""
    kv = {}
    queries = {}
    hidden = {}
    for layer_name in LAYERS:
        layer_idx = int(layer_name.split(".")[2])
        k = torch.randn(n_tokens, KV_HEADS, HEAD_DIM)
        v = torch.randn(n_tokens, KV_HEADS, HEAD_DIM)
        q = torch.randn(n_tokens, HEADS, HEAD_DIM)
        h = torch.randn(n_tokens, HIDDEN)
        kv[layer_idx] = (k, v)
        queries[layer_idx] = q
        hidden[layer_idx] = h
    return kv, queries, hidden


def _reference_attention(q_new, k_visible, v_visible, num_heads, num_kv_heads, head_dim):
    """softmax(q K^T / sqrt(d)) V with GQA repeat."""
    groups = num_heads // num_kv_heads
    k = k_visible.repeat_interleave(groups, dim=1) if groups > 1 else k_visible
    v = v_visible.repeat_interleave(groups, dim=1) if groups > 1 else v_visible
    attn = torch.matmul(q_new, k.transpose(-1, -2)) / (head_dim ** 0.5)
    attn = torch.softmax(attn, dim=-1, dtype=torch.float32)
    return torch.matmul(attn, v)


def _attention_via_metadata(runner, cache: CacheSim, record, row_idx, seq_len_compressed, q_new):
    """Attention computed the way the ascend kernel sees it."""
    bt = runner.input_batch.block_table
    row = bt.block_table.np[row_idx]
    slots = np.concatenate([row[i // BS] * BS + np.arange(BS) for i in range(0, seq_len_compressed, BS)])[
        :seq_len_compressed
    ]
    k_vis, v_vis = cache.read(slots)  # (seq', kv_heads, hd)
    k_vis = k_vis.unsqueeze(0).transpose(1, 2)  # (1, kv_heads, seq', hd)
    v_vis = v_vis.unsqueeze(0).transpose(1, 2)
    q = q_new.unsqueeze(1).unsqueeze(0)  # (1, heads, 1, hd)
    return _reference_attention(q, k_vis, v_vis, HEADS, KV_HEADS, HEAD_DIM)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for key in (
        "KVPRESS",
        "KVPRESS_ASCEND",
        "KVPRESS_ASCEND_ENABLED",
        "KVPRESS_ASCEND_PRESS",
        "KVPRESS_ASCEND_PREFIX_CACHE",
        "KVPRESS_ASCEND_DRY_RUN",
        "KVPRESS_ASCEND_POLICY",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("KVPRESS", "1")


# --------------------------------------------------------------------------- #
# env gating
# --------------------------------------------------------------------------- #


def test_env_gating(monkeypatch):
    monkeypatch.delenv("KVPRESS", raising=False)
    monkeypatch.delenv("KVPRESS_ASCEND", raising=False)
    assert envs.enabled() is False
    monkeypatch.setenv("kvpress", "1")
    assert envs.enabled() is True
    monkeypatch.setenv("KVPRESS_ASCEND", "true")
    assert envs.enabled() is True


def test_press_build():
    p = presses.build_press("snapkv", 0.5, window=32)
    assert p.name == "snapkv" and p.needs_queries
    p = presses.build_press("streamingllm", 0.4, sink=4)
    assert p.name == "streamingllm" and not p.needs_queries
    with pytest.raises(ValueError):
        presses.build_press("nope", 0.5)


# --------------------------------------------------------------------------- #
# patch application against fake modules
# --------------------------------------------------------------------------- #


def test_patch_application():
    install_fake_vllm()
    try:
        engine_mod._PATCHED = False
        engine = engine_mod.apply()
        assert engine is not None
        import vllm_ascend.worker.model_runner_v1 as mr_mod
        import vllm_ascend.worker.block_table as bt_mod
        import vllm_ascend.attention.attention_v1 as attn_v1

        assert getattr(mr_mod.NPUModelRunner.execute_model, "_kvpress_wrapped", False)
        assert getattr(mr_mod.NPUModelRunner._prepare_inputs, "_kvpress_wrapped", False)
        assert getattr(mr_mod.NPUModelRunner._build_attention_metadata, "_kvpress_wrapped", False)
        assert getattr(bt_mod.BlockTable.compute_slot_mapping, "_kvpress_wrapped", False)
        assert getattr(attn_v1.AscendAttentionBackendImpl.forward, "_kvpress_wrapped", False)
        assert getattr(attn_v1.AscendC8AttentionBackendImpl.forward, "_kvpress_wrapped", False)
    finally:
        uninstall_fake_vllm()


def test_patch_status_probes_all_seams():
    """The heartbeat's seam probe must report every patch as installed."""
    install_fake_vllm()
    try:
        engine_mod._PATCHED = False
        engine = engine_mod.apply()
        status = engine.patch_status()
        assert len(status) == len(Engine.SEAM_PROBES)
        assert all(v is True for v in status.values()), status
    finally:
        uninstall_fake_vllm()


def test_enum_attn_state_matches_real_backend():
    """Regression for the on-machine crash: AscendAttentionState is an
    enum.Enum whose .value is an INT (0-4), not a string. The query capture
    must still fire on the real backend (state.name == 'ChunkedPrefill'),
    otherwise snapkv gets queries=None and dies with:

        AttributeError: 'NoneType' object has no attribute 'shape'
    """
    import enum

    class AscendAttentionState(enum.Enum):
        PrefillNoCache = 0
        PrefillCacheHit = 1
        DecodeOnly = 2
        ChunkedPrefill = 3
        SpecDecoding = 4

    engine = Engine(registry=registry_mod.Registry())
    press = presses.build_press("snapkv", 0.5, window=64)
    runner = _make_runner()
    caches = _make_caches()
    _register_caches(runner, caches)
    scheduler = FakeScheduler(BS)
    engine.press = press
    engine.min_len = 0
    runner.set_requests(["req0"], np.array([0], dtype=np.int64), np.array([300], dtype=np.int64))
    kv, q, h = _random_chunk(None, 300)
    meta = FakeAttnMeta(attn_state=AscendAttentionState.ChunkedPrefill)
    simulate_step(runner, engine, scheduler, caches, ["req0"], np.array([300]), kv, q, h, LAYERS, meta)
    record = engine.registry.get("req0")
    assert record is not None, "request must be compressed with Enum attn_state"
    assert 0 in record.keep_indices
    assert record.keep_indices[0].shape[1] == 150, "query capture must have fired"


def test_snapkv_score_falls_back_without_queries():
    """queries=None must not crash the scorer (defensive fallback path)."""
    press = presses.build_press("snapkv", 0.5, window=64)
    keys = torch.randn(1, 2, 300, 16)
    scores = press.score(0, keys, keys, None, None)
    assert scores.shape == (1, 2, 300)
    assert torch.all(scores == 1.0)
    tova = presses.build_press("tova", 0.5)
    scores = tova.score(0, keys, keys, None, None)
    assert scores.shape == (1, 2, 300)


def test_heartbeat_logs_every_step():
    """Every inference step emits one heartbeat line with seam probes and the
    core parameters of the active press."""
    import io
    import logging

    buf = io.StringIO()
    handler = logging.StreamHandler(buf)
    logger = logging.getLogger("kvpress_ascend")
    logger.addHandler(handler)
    try:
        engine = Engine(registry=registry_mod.Registry())
        press = presses.build_press("streamingllm", 0.5, sink=4)
        runner, caches, scheduler, orig_len, record, _ = _compress_with(engine, press)
        handler.flush()
        out = buf.getvalue()
    finally:
        logger.removeHandler(handler)
    assert "step=" in out and "seams=" in out and "press=streamingllm" in out
    assert "params=" in out and "ratio=0.500" in out and "sink=4" in out
    assert "records=" in out
    # heartbeat must appear once per step (two chunked-prefill steps)
    assert out.count("step=") >= 2


# --------------------------------------------------------------------------- #
# end-to-end: chunked prefill -> compress (snapkv) -> decode
# --------------------------------------------------------------------------- #


def _run_prefill_and_compress(engine: Engine, press: Press, prefix_caching=False, dry_run=False, min_len=0):
    runner = _make_runner(prefix_caching=prefix_caching)
    caches = _make_caches()
    _register_caches(runner, caches)
    scheduler = FakeScheduler(BS)
    engine.press = press
    engine.dry_run = dry_run
    engine.min_len = min_len
    engine.registry.records.clear()
    engine.registry.stats.clear()

    orig_len = 300
    rng = np.random.default_rng(0)

    # chunked prefill: 128 + 172 tokens
    kv1, q1, h1 = _random_chunk(rng, 128)
    kv2, q2, h2 = _random_chunk(rng, 172)
    runner.set_requests(["req0"], np.array([0], dtype=np.int64), np.array([orig_len], dtype=np.int64))
    meta = FakeAttnMeta()
    simulate_step(runner, engine, scheduler, caches, ["req0"], np.array([128]), kv1, q1, h1, LAYERS, meta)
    simulate_step(runner, engine, scheduler, caches, ["req0"], np.array([172]), kv2, q2, h2, LAYERS, meta)

    # ground-truth prompt K/V of layer 0 (the exact tensors the sim wrote)
    ref_k0 = torch.cat([kv1[0][0], kv2[0][0]], dim=0)  # (300, kv_heads, hd)
    ref_v0 = torch.cat([kv1[0][1], kv2[0][1]], dim=0)
    return runner, caches, scheduler, orig_len, (ref_k0, ref_v0)


def _compress_with(engine, press, prefix_caching=False, dry_run=False, min_len=0):
    runner, caches, scheduler, orig_len, ref = _run_prefill_and_compress(engine, press, prefix_caching, dry_run, min_len)
    record = engine.registry.get("req0")
    return runner, caches, scheduler, orig_len, record, ref


def test_snapkv_end_to_end_attention_matches_reference():
    engine = Engine(registry=registry_mod.Registry())
    press = presses.build_press("snapkv", 0.5, window=64)
    runner, caches, scheduler, orig_len, record, ref = _compress_with(engine, press)

    assert record is not None, "request should be compressed"
    assert record.n_kept == 150
    assert engine.registry.stats["compressed"] == 1

    # the kept region of the layer-0 cache must contain the top-150 keys
    row = runner.input_batch.block_table.block_table.np[0]
    keep = record.keep_indices[0]
    cache0 = caches[0]
    flat = cache0.key_cache.reshape(-1, KV_HEADS, HEAD_DIM)
    ref_k, _ = ref  # ground truth (300, kv_heads, hd) written during prefill
    m = record.n_blocks_orig
    k = record.n_blocks_kept
    tail_blocks = row[m - k : m]
    kept_slots = np.concatenate([tail_blocks[i] * BS + np.arange(BS) for i in range(k)])[: record.n_kept]
    cache_k = flat[torch.from_numpy(kept_slots)]  # (n_kept, kv_heads, hd)
    for h in range(KV_HEADS):
        assert torch.allclose(
            cache_k[:, h, :],
            ref_k[torch.from_numpy(keep[h]), h, :],
            atol=1e-6,
        ), f"head {h}: kept cache content mismatch"

    # decode step: one new token; attention over the compressed layout must
    # equal attention over (kept tokens + new token)
    rng = np.random.default_rng(1)
    kv3, q3, h3 = _random_chunk(rng, 1)
    meta = FakeAttnMeta()
    simulate_step(runner, engine, scheduler, caches, ["req0"], np.array([1]), kv3, q3, h3, LAYERS, meta)

    q_new = q3[0][0]  # layer 0 query of the new token (heads, hd)
    # reference: kept keys (layer 0) + new key
    cache_v = cache0.value_cache.reshape(-1, KV_HEADS, HEAD_DIM)[torch.from_numpy(kept_slots)]
    k_ref = torch.cat([cache_k, kv3[0][0]], dim=0)  # (n_kept+1, kv_heads, hd)
    v_ref = torch.cat([cache_v, kv3[0][1]], dim=0)  # layer 0 value
    k_ref = k_ref.unsqueeze(0).transpose(1, 2)
    v_ref = v_ref.unsqueeze(0).transpose(1, 2)
    ref_out = _reference_attention(q_new.unsqueeze(1).unsqueeze(0), k_ref, v_ref, HEADS, KV_HEADS, HEAD_DIM)

    # engine view: corrected seq lens + rewritten row
    rec = engine.registry.get("req0")
    assert rec is not None
    seq_compressed = record.n_kept + 1
    eng_out = _attention_via_metadata(runner, cache0, rec, 0, seq_compressed, q_new)
    assert torch.allclose(eng_out, ref_out, atol=1e-4), "attention mismatch between compressed layout and reference"


def test_streamingllm_keep_set():
    engine = Engine(registry=registry_mod.Registry())
    press = presses.build_press("streamingllm", 0.5, sink=4)
    _, _, _, orig_len, record, _ = _compress_with(engine, press)
    assert record is not None
    # streaming-llm: keep first 4 + last (n_kept - 4) tokens, identical across heads
    keep = record.keep_indices[0]
    expected = np.concatenate([np.arange(4), np.arange(orig_len - (150 - 4), orig_len)])
    assert np.array_equal(np.sort(keep[0]), np.sort(expected))


def test_prefix_caching_skips_compression():
    engine = Engine(registry=registry_mod.Registry())
    engine.prefix_cache_mode = "skip"
    press = presses.build_press("streamingllm", 0.5, sink=4)
    runner, cache, scheduler, orig_len, record, _ = _compress_with(engine, press, prefix_caching=True)
    assert record is None, "compression must be skipped with prefix caching"
    assert engine.registry.stats.get("skipped_prefix_cache", 0) == 1


def test_prefix_caching_force_compresses():
    engine = Engine(registry=registry_mod.Registry())
    engine.prefix_cache_mode = "force"
    press = presses.build_press("streamingllm", 0.5, sink=4)
    _, _, _, _, record, _ = _compress_with(engine, press, prefix_caching=True)
    assert record is not None


def test_dry_run_does_not_touch_cache():
    engine = Engine(registry=registry_mod.Registry())
    press = presses.build_press("streamingllm", 0.5, sink=4)
    runner, caches, scheduler, orig_len, record, _ = _compress_with(engine, press, dry_run=True)
    assert record is None
    assert engine.registry.stats.get("dry_run", 0) == 1
    # cache untouched: all blocks still hold their original random content
    row = runner.input_batch.block_table.block_table.np[0]
    assert row.sum() != 0  # row was never rewritten (no record)


def test_short_prompt_skipped():
    engine = Engine(registry=registry_mod.Registry())
    engine.min_len = 100000
    press = presses.build_press("streamingllm", 0.5, sink=4)
    _, _, _, _, record, _ = _compress_with(engine, press, min_len=100000)
    assert record is None
    assert engine.registry.stats.get("skipped_short", 0) == 1


def test_multiple_requests_independent():
    engine = Engine(registry=registry_mod.Registry())
    press = presses.build_press("streamingllm", 0.5, sink=4)
    runner = _make_runner()
    caches = _make_caches(num_blocks=1024)
    _register_caches(runner, caches)
    scheduler = FakeScheduler(BS)
    engine.press = press
    engine.min_len = 0
    runner.set_requests(
        ["reqA", "reqB"],
        np.array([0, 0], dtype=np.int64),
        np.array([300, 200], dtype=np.int64),
    )
    # combined TND batch: reqA tokens first (300), then reqB tokens (200)
    total = 500
    kv, q, h = _random_chunk(None, total)
    meta = FakeAttnMeta()
    simulate_step(runner, engine, scheduler, caches, ["reqA", "reqB"], np.array([300, 200]), kv, q, h, LAYERS, meta)
    assert engine.registry.get("reqA") is not None
    assert engine.registry.get("reqB") is not None
    assert engine.registry.get("reqA").n_kept == 150
    assert engine.registry.get("reqB").n_kept == 100


def test_record_cleanup_on_finish():
    engine = Engine(registry=registry_mod.Registry())
    press = presses.build_press("streamingllm", 0.5, sink=4)
    runner, caches, scheduler, orig_len, record, ref = _compress_with(engine, press)
    assert engine.registry.get("req0") is not None
    from kvpress_ascend.simulate import FakeSchedulerOutput, FakeScheduledReqs

    sched = FakeSchedulerOutput(
        total_num_scheduled_tokens=0,
        num_scheduled_tokens=np.array([0]),
        scheduled_cached_reqs=FakeScheduledReqs(req_ids=["req0"]),
        finished_reqs=["req0"],
    )
    engine.on_execute_model_post(runner, sched, None)
    assert engine.registry.get("req0") is None


def test_per_layer_layout_overrides():
    """SqueezeAttention-style per-layer budgets with per-layer metadata."""
    engine = Engine(registry=registry_mod.Registry())
    press = presses.build_press("streamingllm", 0.5, sink=4)
    runner, caches, scheduler, orig_len, record, ref = _compress_with(engine, press)
    engine.per_layer_mode = True
    # emulate a per-layer budget: layer 0 keeps 150, layer 1 keeps 90
    record.layer_n_kept = {0: 150, 1: 40}
    rec = record

    # next step: metadata built from the (rewritten) scheduler row
    bt = runner.input_batch.block_table
    row = bt.block_table.np[0].copy()
    meta0 = FakeAttnMeta(
        seq_lens=torch.tensor([orig_len + 1], dtype=torch.int64),
        seq_lens_cpu=torch.tensor([orig_len + 1], dtype=torch.int64),
        seq_lens_list=[orig_len + 1],
        block_tables=torch.from_numpy(np.stack([row] * 16)).clone(),
        slot_mapping=torch.zeros(1, dtype=torch.int64),
        actual_seq_lengths_q=[1],
    )
    meta1 = FakeAttnMeta(
        seq_lens=torch.tensor([orig_len + 1], dtype=torch.int64),
        seq_lens_cpu=torch.tensor([orig_len + 1], dtype=torch.int64),
        seq_lens_list=[orig_len + 1],
        block_tables=torch.from_numpy(np.stack([row] * 16)).clone(),
        slot_mapping=torch.zeros(1, dtype=torch.int64),
        actual_seq_lengths_q=[1],
    )
    # positions/query_start_loc for the new token at position orig_len
    runner.positions = torch.tensor([orig_len], dtype=torch.int64)
    runner.query_start_loc.np[:2] = [0, 1]

    from kvpress_ascend.kvcore import per_layer_row

    engine._per_layer_rows["req0"] = {
        0: per_layer_row(rec, row, int(bt.num_blocks_per_row[0]), 150),
        1: per_layer_row(rec, row, int(bt.num_blocks_per_row[0]), 40),
    }
    meta_dict = {"model.layers.0.self_attn.attn": meta0, "model.layers.1.self_attn.attn": meta1}
    engine.on_build_attention_metadata(runner, meta_dict, None)

    assert int(meta0.seq_lens[0]) == 151  # 150 kept + 1 new
    assert int(meta1.seq_lens[0]) == 41  # 40 kept + 1 new
    # per-layer block tables differ
    assert not torch.equal(meta0.block_tables[0], meta1.block_tables[0])
    # layer-0 row front must be the tail blocks of the original row
    m = rec.n_blocks_orig
    k0 = rec.n_blocks_kept
    assert meta0.block_tables[0][0].item() == row[m - k0]


def test_is_prefill_state_all_forms():
    """_is_prefill_state must accept Enum members (real backend, int .value),
    plain strings (offline mocks) and None."""
    import enum

    from kvpress_ascend.engine import _is_prefill_state

    class AscendAttentionState(enum.Enum):
        PrefillNoCache = 0
        PrefillCacheHit = 1
        DecodeOnly = 2
        ChunkedPrefill = 3
        SpecDecoding = 4

    assert _is_prefill_state(AscendAttentionState.PrefillNoCache) is True
    assert _is_prefill_state(AscendAttentionState.PrefillCacheHit) is True
    assert _is_prefill_state(AscendAttentionState.ChunkedPrefill) is True
    assert _is_prefill_state(AscendAttentionState.DecodeOnly) is False
    assert _is_prefill_state(AscendAttentionState.SpecDecoding) is False
    assert _is_prefill_state("ChunkedPrefill") is True
    assert _is_prefill_state("DecodeOnly") is False
    assert _is_prefill_state(None) is False
