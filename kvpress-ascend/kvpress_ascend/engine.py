# SPDX-FileCopyrightText: Copyright (c) 2025 kvpress-ascend contributors
# SPDX-License-Identifier: Apache-2.0
"""Monkeypatch engine adapting KV-cache compression to vllm-ascend (vLLM v1).

All patches target the *worker* side of the vLLM v1 pipeline (the model
runner and the Ascend attention backends).  Nothing in the ``vllm`` /
``vllm_ascend`` packages is modified on disk; every seam is wrapped at
runtime, and every hook fails soft (logs + no-op) so the server keeps serving
even when an API assumption breaks.

Patch points (all verified against vllm-ascend v0.23.0 / vllm 0.23.0):

* ``vllm_ascend.worker.model_runner_v1.NPUModelRunner.execute_model``
    - pre:  snapshot prefill state, publish the per-step CaptureContext
    - post: run the compression pass for requests whose prefill just finished
* ``NPUModelRunner._prepare_inputs``
    - entry: rewrite the block-table rows of compressed requests
* ``vllm_ascend.worker.block_table.BlockTable.compute_slot_mapping`` and
  ``compute_slot_mapping_draft``
    - shift the positions of compressed requests by ``delta`` so new tokens
      are written at slot ``n_kept + j`` of the compressed layout
* ``NPUModelRunner._build_attention_metadata``
    - post: delta-correct ``seq_lens`` of every group-0 layer metadata and of
      the speculative-decode common metadata; apply per-layer block-table /
      slot-mapping overrides (SqueezeAttention layer-wise mode)
* ``vllm_ascend.attention.attention_v1.AscendAttentionBackendImpl.forward``
  (and the C8 subclass)
    - capture the post-RoPE query tail of the final prefill chunk per layer
* ``vllm.model_executor.layers.attention.Attention.forward``
    - capture per-layer input/output hidden states (layer-importance
      measurement used by the SqueezeAttention policy)

The engine is a plain class so the offline simulation can drive the exact
same code paths against mock objects (see ``simulate.py`` / ``tests/``).
"""

from __future__ import annotations

import traceback
from typing import Dict, List, Optional, Tuple

import numpy as np

from . import envs
from .log import get_logger
from .kvcore import (
    compact_layer_inplace,
    compute_scores,
    extract_layer_index,
    finalize_pooled_scores,
    gather_request_kv,
    per_layer_row,
    rewrite_block_table_row,
    select_keep_indices,
    shifted_positions_numpy,
    shifted_positions_tensor,
    split_kv_cache,
)
from .presses import Press, build_press
from .registry import CaptureContext, CompressionRecord, Registry

logger = get_logger()

PREFILL_STATES = ("PrefillNoCache", "PrefillCacheHit", "ChunkedPrefill")


def _synchronize(device) -> None:
    """Block until queued device work completes (needed before cache reads)."""
    try:
        if device is not None and getattr(device, "type", None) == "npu":
            import torch_npu  # noqa: F401

            import torch

            torch.npu.synchronize()
        else:
            import torch

            if torch.cuda.is_available():
                torch.cuda.synchronize()
    except Exception:
        pass


class Engine:
    """Per-process compression engine.  One instance per worker."""

    def __init__(self, press: Optional[Press] = None, registry: Optional[Registry] = None) -> None:
        self.press: Optional[Press] = press
        self.registry = registry if registry is not None else Registry()
        self.window = envs.window()
        self.min_len = envs.min_len()
        self.prefix_cache_mode = envs.prefix_cache_mode()
        self.dry_run = envs.dry_run()
        self.per_layer_mode = False  # enabled by the squeeze policy w/o MTP
        self.capture_hidden = False  # enabled by squeeze policy
        self._last_req_order: List[str] = []
        self._per_layer_rows: Dict[str, Dict[int, np.ndarray]] = {}
        self._prefill_start_computed: Dict[str, int] = {}
        self._warned_no_req_order = False

    # ------------------------------------------------------------------ #
    # step lifecycle (execute_model wrapper)
    # ------------------------------------------------------------------ #

    def on_execute_model_pre(self, runner, scheduler_output) -> None:
        try:
            if not self.press:
                return
            self._per_layer_rows.clear()
            req_ids = self._req_ids(runner)
            num_computed = self._num_computed(runner)
            num_prompt = self._num_prompt(runner)
            num_sched = self._num_scheduled(scheduler_output, req_ids)
            total = getattr(scheduler_output, "total_num_scheduled_tokens", 0) or 0
            is_prefill_step = total > 0 and (
                np.any(np.asarray(num_computed[: len(req_ids)]) < np.asarray(num_prompt[: len(req_ids)]))
                if len(req_ids)
                else False
            )
            ctx = CaptureContext(
                step_id=self.registry.step_counter,
                req_ids=list(req_ids),
                num_scheduled_tokens=np.asarray(num_sched, dtype=np.int64),
                num_computed_tokens_before=np.asarray(num_computed[: len(req_ids)], dtype=np.int64),
                num_prompt_tokens=np.asarray(num_prompt[: len(req_ids)], dtype=np.int64),
                is_prefill_step=bool(is_prefill_step),
            )
            self._last_req_order = list(req_ids)
            # remember the computed-token count at the *start* of prefill for
            # prefix-cache-hit detection
            for i, req_id in enumerate(req_ids):
                if i >= len(num_computed):
                    break
                if ctx.is_prefilling(req_id) and req_id not in self._prefill_start_computed:
                    self._prefill_start_computed[req_id] = int(num_computed[i])
            self.registry.begin_step(ctx)
        except Exception:
            logger.debug("on_execute_model_pre failed:\n%s", traceback.format_exc())

    def on_execute_model_post(self, runner, scheduler_output, output) -> None:
        try:
            if self.press:
                self._maybe_compress(runner, scheduler_output)
                self._drop_finished_records(scheduler_output)
                self.heartbeat(runner)
        except Exception:
            logger.error("kvpress compression pass failed:\n%s", traceback.format_exc())
            self.registry.bump("skipped_error")
        finally:
            self.registry.end_step()

    # ------------------------------------------------------------------ #
    # per-inference patch health heartbeat
    # ------------------------------------------------------------------ #

    SEAM_PROBES = (
        ("vllm_ascend.attention.attention_v1", "AscendAttentionBackendImpl", "forward", "backend_forward"),
        ("vllm_ascend.attention.attention_v1", "AscendC8AttentionBackendImpl", "forward", "backend_c8_forward"),
        ("vllm_ascend.worker.model_runner_v1", "NPUModelRunner", "execute_model", "execute_model"),
        ("vllm_ascend.worker.model_runner_v1", "NPUModelRunner", "_prepare_inputs", "prepare_inputs"),
        ("vllm_ascend.worker.model_runner_v1", "NPUModelRunner", "_build_attention_metadata", "build_attn_metadata"),
        ("vllm_ascend.worker.block_table", "BlockTable", "compute_slot_mapping", "slot_mapping"),
        ("vllm_ascend.worker.block_table", "BlockTable", "compute_slot_mapping_draft", "slot_mapping_draft"),
        ("vllm.model_executor.layers.attention", "Attention", "forward", "attention_module"),
    )

    def patch_status(self) -> dict:
        """Probe every patched seam: True = our wrapper is installed on the
        live method (the patch has really entered the core code)."""
        import importlib

        status = {}
        for module_name, cls_name, method_name, label in self.SEAM_PROBES:
            try:
                cls = getattr(importlib.import_module(module_name), cls_name)
                wrapped = bool(getattr(getattr(cls, method_name), "_kvpress_wrapped", False))
                status[label] = wrapped
            except Exception:
                status[label] = None  # module/class unavailable in this process
        return status

    def heartbeat(self, runner) -> None:
        """One compact log line per inference step: patch seam probes and the
        core parameters of both patches (engine press + squeeze policy)."""
        if not envs.step_log() or self.press is None:
            return
        try:
            status = self.patch_status()
            seam_ok = sum(1 for v in status.values() if v is True)
            seam_total = len(status)
            ok = f"{seam_ok}/{seam_total}"
            if any(v is False for v in status.values()):
                ok += f" !!FAIL:{[k for k, v in status.items() if v is False]}"
            if any(v is None for v in status.values()):
                ok += f" n/a:{[k for k, v in status.items() if v is None]}"

            press = self.press
            params = self._press_params(press)
            req_ids = self._req_ids(runner)
            squeeze = ""
            if press.name == "squeeze":
                squeeze = " squeeze=ACTIVE(capture_hidden=%s)" % self.capture_hidden
                budgets = {}
                for req in req_ids[:8]:
                    b = getattr(press, "_budgets", {}).get(req)
                    if b:
                        budgets[req] = {k: int(v) for k, v in sorted(b.items())}
                if budgets:
                    squeeze += " budgets=" + str(budgets)
            logger.info(
                "step=%d reqs=%d seams=%s records=%d press=%s params=%s%s",
                self.registry.step_counter,
                len(req_ids),
                ok,
                len(self.registry.records),
                press.name,
                params,
                squeeze,
            )
        except Exception:
            logger.debug("heartbeat failed:\n%s", traceback.format_exc())

    @staticmethod
    def _press_params(press) -> str:
        parts = [f"ratio={press.compression_ratio:.3f}"]
        for attr in ("window", "sink", "n_sink", "ini_size", "class3_ratio", "alpha_safeguard"):
            if hasattr(press, attr):
                val = getattr(press, attr)
                if isinstance(val, float):
                    parts.append(f"{attr}={val:.3f}")
                else:
                    parts.append(f"{attr}={val}")
        return " ".join(parts)

    def log_activation_summary(self) -> None:
        """One-time status line after patch installation: every seam probe and
        the core parameters of the active press / squeeze policy."""
        status = self.patch_status()
        seams = " ".join(
            f"{k}={'OK' if v is True else ('N/A' if v is None else 'FAIL')}" for k, v in status.items()
        )
        press = self.press
        squeeze = "inactive"
        if press is not None and press.name == "squeeze":
            squeeze = "active(capture_hidden=%s)" % self.capture_hidden
        logger.info(
            "patch activation summary: seams[%s] press=%s params=[%s] squeeze=%s step_log=%s",
            seams,
            press.name if press is not None else "-",
            self._press_params(press) if press is not None else "-",
            squeeze,
            envs.step_log(),
        )

    # ------------------------------------------------------------------ #
    # compression pass
    # ------------------------------------------------------------------ #

    def _maybe_compress(self, runner, scheduler_output) -> None:
        ctx = self.registry.current()
        if ctx is None or not ctx.is_prefill_step:
            return
        now_computed = self._num_computed(runner)
        prompt = self._num_prompt(runner)
        for i, req_id in enumerate(ctx.req_ids):
            if i >= len(now_computed) or i >= len(prompt):
                break
            if not ctx.is_prefilling(req_id):
                continue
            # NOTE: the input batch's num_computed_tokens is updated in
            # sample_tokens(), i.e. AFTER execute_model returns, so completion
            # must account for the tokens scheduled in the current step.
            after = int(now_computed[i]) + int(ctx.num_scheduled_tokens[i])
            if after < int(prompt[i]):
                continue  # prefill still running (chunked prefill)
            self._compress_one(runner, req_id, i, ctx)

    def _compress_one(self, runner, req_id: str, row_idx: int, ctx: CaptureContext) -> None:
        press = self.press
        if press is None:
            return
        orig_len = int(ctx.num_prompt_tokens[row_idx])
        registry = self.registry
        registry.bump("prefills_completed")

        # --- eligibility -------------------------------------------------- #
        if orig_len < self.min_len:
            registry.bump("skipped_short")
            logger.info("req %s: prompt len %d < KVPRESS_ASCEND_MIN_LEN %d, skip", req_id, orig_len, self.min_len)
            return
        n_kept = int(orig_len * (1 - press.compression_ratio))
        if n_kept < 2:
            registry.bump("skipped_short")
            logger.info("req %s: n_kept=%d too small, skip", req_id, n_kept)
            return
        prefix_cache_on = self._prefix_caching_enabled(runner)
        hit_prefix = ctx.prefill_hit_prefix_cache(req_id) or self._prefill_start_computed.get(req_id, 0) > 0
        if prefix_cache_on and self.prefix_cache_mode != "force":
            registry.bump("skipped_prefix_cache")
            logger.warning(
                "req %s: prefix caching is enabled and KVPRESS_ASCEND_PREFIX_CACHE=skip "
                "(default). Compression skipped for safety. Set "
                "KVPRESS_ASCEND_PREFIX_CACHE=force to compress anyway, or drop "
                "--enable-prefix-caching.",
                req_id,
            )
            return
        if prefix_cache_on and hit_prefix and self.prefix_cache_mode == "force":
            logger.warning(
                "req %s: compressed despite prefix-cache hit (KVPRESS_ASCEND_PREFIX_CACHE=force); "
                "shared prefix blocks are left untouched but future prefix matches on the tail "
                "may be stale.",
                req_id,
            )

        # --- geometry ------------------------------------------------------ #
        try:
            layer_names = self._group0_layer_names(runner)
            if not layer_names:
                logger.warning("req %s: no supported (full-attention) KV group found, skip", req_id)
                return
            bt = self._row_table(runner)
            row = bt.block_table.np[row_idx].copy()
            cur_blocks = int(bt.num_blocks_per_row[row_idx])
            block_size = self._block_size(runner)
            spec = self._group0_spec(runner)
            num_kv_heads = int(spec.num_kv_heads)
            head_dim = int(spec.head_size)
            module0 = self._layer_module(runner, layer_names[0])
            kv_cache = getattr(module0, "kv_cache", None) if module0 is not None else None
            key_cache0, value_cache0 = split_kv_cache(kv_cache)
            if key_cache0 is None:
                logger.warning("req %s: layer kv_cache not bound yet, skip", req_id)
                return
            import torch

            device = key_cache0.device
            dtype = key_cache0.dtype
        except Exception:
            logger.warning("req %s: could not resolve cache geometry, skip:\n%s", req_id, traceback.format_exc())
            registry.bump("skipped_error")
            return

        record = CompressionRecord(
            req_id=req_id,
            orig_len=orig_len,
            n_kept=n_kept,
            block_size=block_size,
            layer_names=layer_names,
        )
        # NOTE: k == m (delta < block_size) is allowed: the block-table row is
        # then an identity rewrite, but the in-place content compaction plus
        # the seq-lens correction still reduce the attended KV length.

        # per-layer budget hooks (SqueezeAttention)
        per_layer = self._use_per_layer_layout(runner)
        try:
            press.prepare(
                layer_names,
                orig_len,
                {"ctx": ctx, "req_id": req_id, "per_layer": per_layer},
            )
            resolved = press.resolved_n_kept(orig_len, req_id)
            if resolved is not None:
                n_kept = int(resolved)
                record.n_kept = n_kept
        except Exception:
            logger.debug("press.prepare failed:\n%s", traceback.format_exc())

        _synchronize(device)
        logger.info(
            "req %s: compressing prompt len=%d -> n_kept=%d (ratio %.3f, k=%d blocks, dry_run=%s)",
            req_id,
            orig_len,
            n_kept,
            press.compression_ratio,
            record.n_blocks_kept,
            self.dry_run,
        )

        pooled_scores: Dict[int, object] = {}
        try:
            if press.pool_across_layers:
                # pass 1: score every layer (no compaction)
                for layer_name in layer_names:
                    layer_idx = extract_layer_index(layer_name)
                    key_cache, value_cache = self._layer_caches(runner, layer_name)
                    keys, values = gather_request_kv(
                        key_cache, value_cache, row, orig_len, block_size, num_kv_heads, head_dim, device, dtype
                    )
                    queries = self._queries_for(ctx, req_id, layer_idx)
                    compute_scores(press, layer_idx, keys, values, queries, None, pooled_scores, extra={"ctx": ctx, "req_id": req_id})
                pooled = finalize_pooled_scores(press, pooled_scores, device)
                keep = select_keep_indices(pooled, n_kept, device)
                record.keep_indices = {extract_layer_index(n): keep.copy() for n in layer_names}
                if not self.dry_run:
                    for layer_name in layer_names:
                        key_cache, value_cache = self._layer_caches(runner, layer_name)
                        compact_layer_inplace(
                            key_cache, value_cache, row, record, keep, num_kv_heads, head_dim, device, dtype
                        )
            else:
                # per-layer scoring + compaction, streamed layer by layer
                for layer_name in layer_names:
                    layer_idx = extract_layer_index(layer_name)
                    n_kept_l = press.n_kept_layer(layer_idx, orig_len, req_id) or n_kept
                    n_kept_l = max(min(int(n_kept_l), orig_len - 1), 2)
                    key_cache, value_cache = self._layer_caches(runner, layer_name)
                    keys, values = gather_request_kv(
                        key_cache, value_cache, row, orig_len, block_size, num_kv_heads, head_dim, device, dtype
                    )
                    queries = self._queries_for(ctx, req_id, layer_idx)
                    scores = press.score(
                        layer_idx, keys, values, queries, None, {"ctx": ctx, "req_id": req_id}
                    )
                    keep = select_keep_indices(scores, n_kept_l, device)
                    record.keep_indices[layer_idx] = keep
                    record.layer_n_kept[layer_idx] = n_kept_l
                    compact_layer_inplace(
                        key_cache,
                        value_cache,
                        row,
                        record,
                        keep,
                        num_kv_heads,
                        head_dim,
                        device,
                        dtype,
                        self.dry_run,
                        n_kept_override=n_kept_l,
                    )
        except Exception:
            logger.error("req %s: compression failed, request left uncompressed:\n%s", req_id, traceback.format_exc())
            registry.bump("skipped_error")
            return

        if self.dry_run:
            registry.bump("dry_run")
            logger.info("req %s: [dry-run] compression computed, cache untouched", req_id)
            return
        registry.put(record)
        registry.bump("compressed")
        logger.info("req %s: compressed ok (n_kept=%d)", req_id, n_kept)

    # ------------------------------------------------------------------ #
    # _prepare_inputs entry: block-table row rewrite
    # ------------------------------------------------------------------ #

    def on_prepare_inputs_entry(self, runner) -> None:
        try:
            if not self.press or not self.registry.records:
                return
            req_ids = self._req_ids(runner)
            bt = self._row_table(runner)
            if bt is None:
                return
            row_np = bt.block_table.np
            num_blocks_per_row = bt.num_blocks_per_row
            row_width = row_np.shape[1]
            for i, req_id in enumerate(req_ids):
                record = self.registry.get(req_id)
                if record is None:
                    continue
                cur_blocks = int(num_blocks_per_row[i])
                if self.per_layer_mode:
                    layer_rows: Dict[int, np.ndarray] = {}
                    for layer_name in record.layer_names:
                        layer_idx = extract_layer_index(layer_name)
                        n_kept_l = self._n_kept_layer(record, layer_idx)
                        layer_rows[layer_idx] = per_layer_row(record, row_np[i], cur_blocks, n_kept_l)
                    self._per_layer_rows[req_id] = layer_rows
                rewrite_block_table_row(row_np[i], row_width, cur_blocks, record)
        except Exception:
            logger.debug("on_prepare_inputs_entry failed:\n%s", traceback.format_exc())

    # ------------------------------------------------------------------ #
    # slot-mapping position shift
    # ------------------------------------------------------------------ #

    def on_compute_slot_mapping(self, num_reqs: int, query_start_loc, positions):
        """Return the (possibly shifted) positions tensor for slot mapping."""
        try:
            if not self.registry.records:
                return positions
            req_order = self._last_req_order[:num_reqs]
            return shifted_positions_tensor(positions, query_start_loc, req_order, self.registry, positions.device)
        except Exception:
            if not self._warned_no_req_order:
                self._warned_no_req_order = True
                logger.warning("compute_slot_mapping shift failed, using unshifted positions:\n%s", traceback.format_exc())
            return positions

    def on_compute_slot_mapping_draft(self, req_indices, positions):
        """numpy variant used for speculative-draft slot mapping."""
        try:
            if not self.registry.records:
                return positions
            return shifted_positions_numpy(req_indices, positions, self._last_req_order, self.registry)
        except Exception:
            return positions

    # ------------------------------------------------------------------ #
    # attention-metadata correction
    # ------------------------------------------------------------------ #

    def on_build_attention_metadata(self, runner, attn_metadata, spec_decode_common_attn_metadata):
        try:
            if not self.press or not self.registry.records:
                return attn_metadata, spec_decode_common_attn_metadata
            req_ids = self._req_ids(runner)
            num_reqs = len(req_ids)
            group0 = self._group0_layer_names(runner)

            meta_entries = self._iter_metadata_entries(attn_metadata)
            for layer_name, meta in meta_entries:
                if layer_name not in group0:
                    continue
                self._correct_meta(runner, meta, req_ids[:num_reqs], layer_name)
            if spec_decode_common_attn_metadata is not None:
                self._correct_common_metadata(spec_decode_common_attn_metadata, req_ids[:num_reqs])
        except Exception:
            logger.debug("on_build_attention_metadata failed:\n%s", traceback.format_exc())
        return attn_metadata, spec_decode_common_attn_metadata

    def _correct_meta(self, runner, meta, req_ids: List[str], layer_name: str) -> None:
        layer_idx = extract_layer_index(layer_name)
        recs = [(i, self.registry.get(req)) for i, req in enumerate(req_ids)]
        recs = [(i, r) for i, r in recs if r is not None]
        if not recs:
            return

        # ---- seq lens --------------------------------------------------- #
        seq = getattr(meta, "seq_lens", None)
        if seq is not None:
            import torch

            corrected = seq.clone()
            for i, rec in recs:
                corrected[i] = corrected[i] - (rec.orig_len - self._n_kept_layer(rec, layer_idx))
            meta.seq_lens = corrected
            meta.seq_lens_cpu = corrected
            seq_list = getattr(meta, "seq_lens_list", None)
            if isinstance(seq_list, list):
                for i, rec in recs:
                    if i < len(seq_list):
                        seq_list[i] = seq_list[i] - (rec.orig_len - self._n_kept_layer(rec, layer_idx))

        # ---- per-layer layout overrides (squeeze, no spec decode) ------- #
        if self.per_layer_mode:
            self._override_per_layer_layout(runner, meta, recs, layer_idx)

    def _override_per_layer_layout(self, runner, meta, recs, layer_idx: int) -> None:
        import torch

        bt = self._row_table(runner)
        if bt is None:
            return
        req_ids_of_rows = self._req_ids(runner)
        row_np = bt.block_table.np
        block_size = self._block_size(runner)

        block_tables = getattr(meta, "block_tables", None)
        if block_tables is None:
            return
        new_bt = block_tables.clone()
        # per-layer slot mapping over the whole token batch
        slot_mapping = getattr(meta, "slot_mapping", None)
        new_slots = slot_mapping.clone() if slot_mapping is not None else None
        positions = getattr(runner, "positions", None)
        if positions is not None:
            positions = getattr(positions, "gpu", positions)

        for i, rec in recs:
            layer_rows = self._per_layer_rows.get(rec.req_id)
            if layer_rows is None or layer_idx not in layer_rows:
                continue
            row_idx = self._row_index(runner, rec.req_id)
            if row_idx is None:
                continue
            row_l = layer_rows[layer_idx]
            new_bt[row_idx] = torch.from_numpy(row_l).to(block_tables.device)
            # per-layer slot mapping: slot = row_l[p' // bs] * bs + p' % bs
            if new_slots is not None and positions is not None:
                qsl = getattr(runner, "query_start_loc", None)
                qsl_np = getattr(qsl, "np", qsl)
                start = int(qsl_np[i]) if qsl_np is not None else None
                end = int(qsl_np[i + 1]) if qsl_np is not None else None
                if start is not None:
                    p_orig = positions[start:end].clone()
                    n_kept_l = self._n_kept_layer(rec, layer_idx)
                    p = p_orig - (rec.orig_len - n_kept_l)
                    block_idx = p // block_size
                    off = p % block_size
                    row_t = torch.from_numpy(row_l).to(p.device)
                    slots = row_t[block_idx.long()] * block_size + off
                    new_slots[start:end] = slots.long()
        meta.block_tables = new_bt
        if new_slots is not None:
            meta.slot_mapping = new_slots

    def _correct_common_metadata(self, cm, req_ids: List[str]) -> None:
        recs = [(i, self.registry.get(req)) for i, req in enumerate(req_ids)]
        recs = [(i, r) for i, r in recs if r is not None]
        if not recs:
            return
        import torch

        for attr in ("seq_lens", "seq_lens_cpu", "_seq_lens_cpu"):
            seq = getattr(cm, attr, None)
            if seq is None:
                continue
            corrected = seq.clone()
            for i, rec in recs:
                corrected[i] = corrected[i] - rec.delta
            setattr(cm, attr, corrected)

    # ------------------------------------------------------------------ #
    # backend capture: query tails (SnapKV/TOVA/...) and hidden states
    # ------------------------------------------------------------------ #

    def on_backend_forward(self, layer, query, attn_metadata, is_draft: bool = False) -> None:
        try:
            ctx = self.registry.current()
            if ctx is None or not ctx.is_prefill_step or is_draft:
                return
            press = self.press
            if press is None or not (press.needs_queries or self.capture_hidden):
                return
            state = getattr(attn_metadata, "attn_state", None)
            state = getattr(state, "value", state)  # Enum support
            if state not in PREFILL_STATES:
                return
            layer_name = getattr(layer, "layer_name", None)
            if not layer_name:
                return
            layer_idx = extract_layer_index(layer_name)
            q_lens = getattr(attn_metadata, "actual_seq_lengths_q", None)
            if q_lens is None or query is None:
                return
            start = 0
            for i, req_id in enumerate(ctx.req_ids):
                if i >= len(q_lens):
                    break
                q_len = int(q_lens[i])
                if q_len <= 0:
                    continue
                if ctx.is_prefilling(req_id):
                    cap = ctx.captures.setdefault(req_id, ReqCapture())
                    if press.needs_queries:
                        w = min(self.window, q_len)
                        # query is TND (T, heads, hd); keep the last w tokens
                        q_t = query.transpose(0, 1)  # (heads, T, hd)
                        cap.queries[layer_idx] = q_t[:, start + q_len - w : start + q_len, :].unsqueeze(0)
                start += q_len
        except Exception:
            logger.debug("on_backend_forward capture failed:\n%s", traceback.format_exc())

    def on_attention_module_forward(self, layer, hidden_states, attn_metadata, out) -> None:
        """Capture per-layer input/output hidden states (squeeze importance)."""
        try:
            if not self.capture_hidden:
                return
            ctx = self.registry.current()
            if ctx is None or not ctx.is_prefill_step:
                return
            state = getattr(attn_metadata, "attn_state", None)
            state = getattr(state, "value", state)
            if state not in PREFILL_STATES:
                return
            layer_name = getattr(layer, "layer_name", None)
            if not layer_name or hidden_states is None or out is None:
                return
            layer_idx = extract_layer_index(layer_name)
            q_lens = getattr(attn_metadata, "actual_seq_lengths_q", None)
            if q_lens is None:
                return
            import torch
            import torch.nn.functional as F

            start = 0
            for i, req_id in enumerate(ctx.req_ids):
                if i >= len(q_lens):
                    break
                q_len = int(q_lens[i])
                if q_len <= 0:
                    continue
                if ctx.is_prefilling(req_id):
                    hs = hidden_states[start : start + q_len]
                    os_ = out[start : start + q_len]
                    cos = float(F.cosine_similarity(hs, os_, dim=-1).mean().detach().cpu())
                    cap = ctx.captures.setdefault(req_id, ReqCapture())
                    cap.cos_sims.setdefault(layer_idx, []).append(cos)
                start += q_len
        except Exception:
            logger.debug("on_attention_module_forward capture failed:\n%s", traceback.format_exc())

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #

    def set_press(self, press: Press, per_layer_mode: bool = False, capture_hidden: bool = False, force: bool = False) -> None:
        if self.press is not None and self.press.name != press.name:
            if not force:
                logger.warning(
                    "press %r already active; press %r ignored (set KVPRESS_ASCEND_POLICY to prefer it)",
                    self.press.name,
                    press.name,
                )
                return
            logger.warning("press %r replaced by %r (KVPRESS_ASCEND_POLICY)", self.press.name, press.name)
        self.press = press
        self.per_layer_mode = per_layer_mode
        self.capture_hidden = capture_hidden or press.needs_hidden

    def _queries_for(self, ctx: CaptureContext, req_id: str, layer_idx: int):
        cap = ctx.captures.get(req_id)
        if cap is None:
            return None
        return cap.queries.get(layer_idx)

    def _n_kept_layer(self, record: CompressionRecord, layer_idx: int) -> int:
        layer_n_kept = getattr(record, "layer_n_kept", None)
        if layer_n_kept:
            return int(layer_n_kept.get(layer_idx, record.n_kept))
        return record.n_kept

    def _use_per_layer_layout(self, runner) -> bool:
        """Per-layer layouts are incompatible with speculative decoding (the
        draft model shares the group-0 cache with one fixed layout)."""
        if not self.per_layer_mode:
            return False
        if getattr(runner, "speculative_config", None) is not None:
            return False
        return True

    def _req_ids(self, runner) -> List[str]:
        return list(getattr(runner.input_batch, "req_ids", []) or [])

    def _num_computed(self, runner) -> np.ndarray:
        ib = runner.input_batch
        for attr in ("num_computed_tokens_cpu", "num_computed_tokens"):
            val = getattr(ib, attr, None)
            if val is not None:
                arr = val.numpy() if hasattr(val, "numpy") else np.asarray(val)
                return arr
        return np.zeros(len(self._req_ids(runner)), dtype=np.int64)

    def _num_prompt(self, runner) -> np.ndarray:
        ib = runner.input_batch
        for attr in ("num_prompt_tokens", "num_prompt_tokens_cpu"):
            val = getattr(ib, attr, None)
            if val is not None:
                arr = val.numpy() if hasattr(val, "numpy") else np.asarray(val)
                return arr
        return np.zeros(len(self._req_ids(runner)), dtype=np.int64)

    def _num_scheduled(self, scheduler_output, req_ids: List[str]) -> np.ndarray:
        ns = getattr(scheduler_output, "num_scheduled_tokens", None)
        if ns is not None and len(ns) >= len(req_ids):
            return np.asarray(ns[: len(req_ids)], dtype=np.int64)
        return np.ones(len(req_ids), dtype=np.int64) * (getattr(scheduler_output, "total_num_scheduled_tokens", 0) or 0)

    def _row_table(self, runner):
        ib = runner.input_batch
        bt = getattr(ib, "block_table", None)
        if bt is None:
            return None
        if hasattr(bt, "__getitem__") and hasattr(bt, "block_tables"):
            try:
                return bt[0]
            except Exception:
                return None
        return bt

    def _row_index(self, runner, req_id: str):
        ib = runner.input_batch
        mapping = getattr(ib, "req_id_to_index", None)
        if mapping is not None and req_id in mapping:
            return mapping[req_id]
        return None

    def _group0_layer_names(self, runner) -> List[str]:
        groups = getattr(getattr(runner, "kv_cache_config", None), "kv_cache_groups", None) or []
        if not groups:
            return []
        g0 = groups[0]
        spec = getattr(g0, "kv_cache_spec", None)
        spec_cls = type(spec).__name__
        if "FullAttention" not in spec_cls and "Attention" not in spec_cls:
            logger.warning(
                "kvpress-ascend currently supports plain full-attention KV groups; "
                "group 0 spec is %s. Skipping compression.",
                spec_cls,
            )
            return []
        return list(getattr(g0, "layer_names", []) or [])

    def _group0_spec(self, runner):
        groups = getattr(getattr(runner, "kv_cache_config", None), "kv_cache_groups", None) or []
        return groups[0].kv_cache_spec if groups else None

    def _block_size(self, runner) -> int:
        spec = self._group0_spec(runner)
        bs = getattr(spec, "block_size", None)
        if bs:
            return int(bs)
        bt = self._row_table(runner)
        if bt is not None:
            return int(getattr(bt, "block_size", 128))
        return 128

    def _layer_module(self, runner, layer_name: str):
        fc = getattr(getattr(runner, "compilation_config", None), "static_forward_context", None)
        if fc is None:
            fc = getattr(runner, "forward_context", None)
        if fc is not None and layer_name in fc:
            return fc[layer_name]
        return None

    def _layer_caches(self, runner, layer_name: str):
        """(key_cache, value_cache) of one layer, resolved per layer."""
        module = self._layer_module(runner, layer_name)
        kv_cache = getattr(module, "kv_cache", None) if module is not None else None
        return split_kv_cache(kv_cache)

    def _prefix_caching_enabled(self, runner) -> bool:
        cache_cfg = getattr(runner, "cache_config", None)
        if cache_cfg is None:
            cc = getattr(getattr(runner, "vllm_config", None), "cache_config", None)
            cache_cfg = cc
        if cache_cfg is not None:
            return bool(getattr(cache_cfg, "enable_prefix_caching", False))
        return True  # be conservative when unknown

    def _iter_metadata_entries(self, attn_metadata):
        """Yield (layer_name, meta) from dict or list-of-dicts metadata."""
        if isinstance(attn_metadata, dict):
            for name, meta in attn_metadata.items():
                if meta is not None:
                    yield name, meta
        elif isinstance(attn_metadata, (list, tuple)):
            for sub in attn_metadata:
                if isinstance(sub, dict):
                    for name, meta in sub.items():
                        if meta is not None:
                            yield name, meta

    def _drop_finished_records(self, scheduler_output) -> None:
        finished = getattr(scheduler_output, "finished_reqs", None)
        if not finished:
            return
        for req in finished:
            req_id = getattr(req, "req_id", None) or (req if isinstance(req, str) else None)
            if req_id is not None:
                self.registry.drop(req_id)
                self._prefill_start_computed.pop(req_id, None)


class ReqCapture:
    """Per-request capture buffer accumulated across prefill chunks."""

    __slots__ = ("queries", "cos_sims")

    def __init__(self) -> None:
        self.queries: Dict[int, object] = {}
        self.cos_sims: Dict[int, List[float]] = {}


# --------------------------------------------------------------------------- #
# Patch application (runtime)
# --------------------------------------------------------------------------- #

_PATCHED = False


def _defuse_vllm_ascend_imports() -> bool:
    """Defuse vllm-ascend's latent circular import before anything else.

    vllm-ascend v0.23.0 has an order-sensitive module cycle::

        vllm_ascend/ops/__init__.py:21  import ...ops.fused_moe.fused_moe
        fused_moe.py:41                 from ...experts_selector import select_experts
        experts_selector.py:25          from vllm_ascend.device.device_op import DeviceOperator
        device_op.py:32                 from vllm_ascend.ops.triton.fla.chunk_scaled_dot_kkt import ...

    The natural import order (the one vllm's CLI uses) completes the cycle
    cleanly because the triton/fla path has no back-edge.  Importing the cycle
    from the *wrong* entry (e.g. ``device_op`` or ``experts_selector`` first,
    exactly what this package's own pre-import of ``attention_v1`` used to do)
    fails mid-cycle, and the swallowed failure leaves partial modules in
    ``sys.modules`` that break vllm's own CLI import chain afterwards
    ("cannot import name 'select_experts' from partially initialized module
    '...experts_selector'").

    Fix: at activation, import the cycle through its canonical safe entry
    FIRST.  All cycle modules then finish initializing in the verified-safe
    order and stay cached, so every later import order (vllm CLI, engine
    core, workers) hits only complete modules and the latent cycle can never
    fire again.

    Returns True when the defuse succeeded; on failure we abort patching
    rather than risk leaving partial modules behind.
    """
    try:
        import vllm_ascend.ops.fused_moe.fused_moe  # noqa: F401
        return True
    except Exception as exc:
        logger.error(
            "vllm-ascend import-cycle defuse failed (%s). Aborting patch installation "
            "to avoid leaving partially-initialized modules. Fallback activation: "
            "export VLLM_PLUGINS=kvpress_ascend (lets vllm import the packages in its "
            "own order).",
            exc,
        )
        return False


def apply(force: bool = False) -> Engine:
    """Create the engine and install all monkeypatches.

    Safe to call multiple times (idempotent).  Raises nothing: any failure is
    logged and the server keeps running without compression.
    """
    global _PATCHED
    if _PATCHED and not force:
        return _get_engine()
    engine = _get_engine()
    if engine.press is None:
        engine.press = build_press(envs.press(), envs.ratio(), window=envs.window(), sink=envs.sink())
        logger.info(
            "kvpress-ascend enabled: press=%s ratio=%.3f window=%d dry_run=%s",
            engine.press.name,
            engine.press.compression_ratio,
            engine.window,
            engine.dry_run,
        )
    # MUST run before any other vllm_ascend import: our own pre-import of
    # attention_v1 would otherwise enter vllm-ascend's latent circular import
    # from the wrong side and break vllm's CLI startup (see the defuse docstring).
    if not _defuse_vllm_ascend_imports():
        return engine
    try:
        _patch_vllm_ascend(engine)
        _PATCHED = True
        engine.log_activation_summary()
        return engine
    except Exception:
        logger.error("kvpress-ascend patch installation failed (server continues unpatched):\n%s", traceback.format_exc())
        return engine


_ENGINE: Optional[Engine] = None


def _get_engine() -> Engine:
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = Engine()
    return _ENGINE


def get_engine() -> Engine:
    return _get_engine()


def _patch_vllm_ascend(engine: Engine) -> None:
    """Import the vllm/vllm_ascend classes and wrap the seams."""
    import vllm.model_executor.layers.attention as vllm_attn_mod  # noqa: F401
    import vllm_ascend.attention.attention_v1 as attn_v1
    import vllm_ascend.worker.block_table as bt_mod
    import vllm_ascend.worker.model_runner_v1 as mr_mod

    # --- 1. backend forward: capture query tails -------------------------- #
    for cls_name in ("AscendAttentionBackendImpl", "AscendC8AttentionBackendImpl"):
        cls = getattr(attn_v1, cls_name, None)
        if cls is None or not hasattr(cls, "forward"):
            continue
        orig = cls.forward
        if getattr(orig, "_kvpress_wrapped", False):
            continue

        def _make_backend_wrapper(orig_forward):
            def wrapper(self, layer, query, key=None, value=None, kv_cache=None, attn_metadata=None, **kwargs):
                is_draft = False
                try:
                    from vllm_ascend.ascend_forward_context import _EXTRA_CTX

                    is_draft = bool(getattr(_EXTRA_CTX, "is_draft_model", False))
                except Exception:
                    pass
                engine.on_backend_forward(layer, query, attn_metadata, is_draft=is_draft)
                return orig_forward(self, layer, query, key, value, kv_cache, attn_metadata, **kwargs)

            wrapper._kvpress_wrapped = True
            return wrapper

        cls.forward = _make_backend_wrapper(orig)
        logger.debug("patched %s.forward", cls_name)

    # --- 2. Attention module forward: hidden-state capture (squeeze) ----- #
    attn_cls = getattr(vllm_attn_mod, "Attention", None)
    if attn_cls is not None and hasattr(attn_cls, "forward"):
        orig_attn = attn_cls.forward
        if not getattr(orig_attn, "_kvpress_wrapped", False):

            def _make_attn_wrapper(orig_forward):
                def wrapper(self, *args, **kwargs):
                    layer = kwargs.get("layer")
                    hidden = kwargs.get("hidden_states")
                    attn_meta = kwargs.get("attn_metadata")
                    if layer is None and len(args) >= 1:
                        layer = args[0]
                    if hidden is None and len(args) >= 2:
                        hidden = args[1]
                    if attn_meta is None and len(args) >= 5:
                        attn_meta = args[4]
                    out = orig_forward(self, *args, **kwargs)
                    if layer is not None and hidden is not None and attn_meta is not None:
                        engine.on_attention_module_forward(layer, hidden, attn_meta, out)
                    return out

                wrapper._kvpress_wrapped = True
                return wrapper

            attn_cls.forward = _make_attn_wrapper(orig_attn)
            logger.debug("patched vllm Attention.forward")

    # --- 3. NPUModelRunner.execute_model ---------------------------------- #
    runner_cls = getattr(mr_mod, "NPUModelRunner", None)
    if runner_cls is None:
        raise RuntimeError("NPUModelRunner not found in vllm_ascend.worker.model_runner_v1")
    orig_exec = runner_cls.execute_model
    if not getattr(orig_exec, "_kvpress_wrapped", False):

        def _make_exec_wrapper(orig_forward):
            def wrapper(self, scheduler_output, intermediate_tensors=None):
                out = None
                engine.on_execute_model_pre(self, scheduler_output)
                try:
                    out = orig_forward(self, scheduler_output, intermediate_tensors)
                finally:
                    engine.on_execute_model_post(self, scheduler_output, out)
                return out

            wrapper._kvpress_wrapped = True
            return wrapper

        runner_cls.execute_model = _make_exec_wrapper(orig_exec)
        logger.debug("patched NPUModelRunner.execute_model")

    # --- 4. NPUModelRunner._prepare_inputs --------------------------------- #
    orig_prep = runner_cls._prepare_inputs
    if not getattr(orig_prep, "_kvpress_wrapped", False):

        def _make_prep_wrapper(orig_forward):
            def wrapper(self, scheduler_output, num_scheduled_tokens):
                engine.on_prepare_inputs_entry(self)
                return orig_forward(self, scheduler_output, num_scheduled_tokens)

            wrapper._kvpress_wrapped = True
            return wrapper

        runner_cls._prepare_inputs = _make_prep_wrapper(orig_prep)
        logger.debug("patched NPUModelRunner._prepare_inputs")

    # --- 5. NPUModelRunner._build_attention_metadata ----------------------- #
    orig_build = runner_cls._build_attention_metadata
    if not getattr(orig_build, "_kvpress_wrapped", False):

        def _make_build_wrapper(orig_forward):
            def wrapper(self, *args, **kwargs):
                out = orig_forward(self, *args, **kwargs)
                if isinstance(out, tuple) and len(out) >= 2:
                    engine.on_build_attention_metadata(self, out[0], out[1])
                return out

            wrapper._kvpress_wrapped = True
            return wrapper

        runner_cls._build_attention_metadata = _make_build_wrapper(orig_build)
        logger.debug("patched NPUModelRunner._build_attention_metadata")

    # --- 6. BlockTable slot-mapping position shift ------------------------- #
    for method_name in ("compute_slot_mapping", "compute_slot_mapping_draft"):
        cls = getattr(bt_mod, "BlockTable", None)
        if cls is None or not hasattr(cls, method_name):
            continue
        orig_m = getattr(cls, method_name)
        if getattr(orig_m, "_kvpress_wrapped", False):
            continue

        if method_name == "compute_slot_mapping":

            def _make_sm_wrapper(orig_forward):
                def wrapper(self, num_reqs, query_start_loc, positions):
                    positions = engine.on_compute_slot_mapping(num_reqs, query_start_loc, positions)
                    return orig_forward(self, num_reqs, query_start_loc, positions)

                wrapper._kvpress_wrapped = True
                return wrapper

        else:

            def _make_smd_wrapper(orig_forward):
                def wrapper(self, req_indices, positions, **kwargs):
                    positions = engine.on_compute_slot_mapping_draft(req_indices, positions)
                    return orig_forward(self, req_indices, positions, **kwargs)

                wrapper._kvpress_wrapped = True
                return wrapper

        setattr(cls, method_name, _make_sm_wrapper(orig_m) if method_name == "compute_slot_mapping" else _make_smd_wrapper(orig_m))
        logger.debug("patched BlockTable.%s", method_name)

    logger.info("kvpress-ascend: all patches installed")
