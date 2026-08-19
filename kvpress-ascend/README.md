# kvpress-ascend

Monkeypatch adapter that ports **NVIDIA [kvpress](https://github.com/NVIDIA/kvpress)** KV-cache
compression to **vllm-ascend** (vLLM v1 engine, Ascend NPU) **without modifying a single line of
the installed `vllm` / `vllm-ascend` packages**.

`kvpress` was built for 🤗 transformers (`model.generate` with dense KV caches and forward hooks).
vLLM v1 runs the model through paged attention backends with a block-managed KV cache, so the
library is re-implemented on top of the vLLM worker seams.  All patches are applied at runtime,
fail soft (the server keeps serving uncompressed if anything goes wrong) and are gated by an
environment variable.

## Quick start (on the NPU machine)

```bash
# 1. install (from this directory, or any checkout of it)
pip install ./kvpress-ascend

# 2. launch vllm-ascend exactly as usual, with one extra export:
export kvpress=1          # activates the patch (also: KVPRESS, KVPRESS_ASCEND, ...)
vllm serve /softwarePlatform/c00879303/Qwen3.5-27B-w8a8-mtp ...   # your normal command
```

The package ships a `kvpress_ascend.pth` file that is installed into site-packages, so *every*
Python process — the API server, the engine-core scheduler and each TP worker — imports
`kvpress_ascend` at startup.  Without `export kvpress=1` the package is completely inert.

## What happens at runtime

1. **Prefill**: the engine watches every prefill step of every request.
2. **At prefill completion** (per request, per layer, on the worker):
   - gathers the request's dense K/V from its paged cache blocks,
   - computes the press scores (kvpress-style, see below),
   - rewrites the request's own *tail blocks* in place so that only the top-`n_kept`
     (key, value) pairs per head remain, densely packed,
   - records a per-request compression record.
3. **Every later step** the engine rewrites the request's *logical* view so the attention
   kernels only ever see the kept tokens:
   - the worker-side block-table row becomes `[kept tail blocks] + [scheduler-grown blocks]`,
   - the slot mapping shifts the request's positions by `delta` (new tokens continue at
     slot `n_kept + j`),
   - the attention metadata `seq_lens` are corrected by `delta` per layer,
   - the speculative-decode (MTP/EAGLE) metadata is corrected the same way, so the draft model
     sees exactly the same compressed view as the target model.

The compression is *per-head* (each KV head keeps its own top-k set), which is exactly
kvpress `ScorerPress` semantics.  Attention cost scales with `n_kept + generated` instead of
`prompt_len + generated`.

## Presses

| `KVPRESS_ASCEND_PRESS` | method | needs queries |
|---|---|---|
| `snapkv` (default) | SnapKV window attention (paper: [2404.14469](https://arxiv.org/abs/2404.14469)) | yes |
| `streamingllm` | sink + recent tokens ([2309.17453](https://arxiv.org/abs/2309.17453)) | no |
| `random` | random pruning (baseline) | no |
| `knorm` | key-norm pruning ([2406.11430](https://arxiv.org/abs/2406.11430)) | no |
| `tova` | last-token attention ([2401.06104](https://arxiv.org/abs/2401.06104)) | yes |
| `pyramidkv` | SnapKV scores pooled across layers ([2409.02054](https://arxiv.org/abs/2409.02054)) | yes |
| `adakv` | per-head top-k with safeguard, AdaKV-style ([2407.11550](https://arxiv.org/abs/2407.11550)) | yes |

Presses that need queries capture the post-RoPE query tail of the final prefill chunk from the
Ascend attention backend (no extra model forward).  `ObservedAttention`-style presses (which
require attention weights from the kernel) are not supported on Ascend.

## Environment variables

| Variable | Default | Meaning |
|---|---|---|
| `kvpress` / `KVPRESS` / `KVPRESS_ASCEND` / `KVPRESS_ASCEND_ENABLED` | – | activate the patch |
| `KVPRESS_ASCEND_PRESS` | `snapkv` | press name (table above) |
| `KVPRESS_ASCEND_RATIO` | `0.5` | compression ratio in `[0,1)` |
| `KVPRESS_ASCEND_WINDOW` | `64` | SnapKV/TOVA observation window |
| `KVPRESS_ASCEND_SINK` | `4` | streaming-LLM sink tokens |
| `KVPRESS_ASCEND_MIN_LEN` | `2048` | do not compress shorter prompts |
| `KVPRESS_ASCEND_PREFIX_CACHE` | `skip` | `skip` = refuse to compress when `--enable-prefix-caching` is on; `force` = compress anyway (you accept stale tail-block hashes in the block pool) |
| `KVPRESS_ASCEND_DRY_RUN` | `0` | `1` = compute scores + log statistics, never touch the cache |
| `KVPRESS_ASCEND_LOG` | `info` | `debug` / `info` / `warning` |
| `KVPRESS_ASCEND_POLICY` | `kvpress` | `kvpress` or `squeeze` when both packages are enabled |
| `VLLM_PLUGINS` | – | alternative activation: `export VLLM_PLUGINS=kvpress_ascend` |

## Important caveats

* **Prefix caching.**  Compression rewrites the physical content of the request's tail blocks.
  With `--enable-prefix-caching`, those blocks are registered in the engine-core block pool and a
  future request matching the same tokens would reuse stale content.  The safe default is
  therefore `KVPRESS_ASCEND_PREFIX_CACHE=skip` — compression is skipped while prefix caching is
  enabled.  For maximum savings either drop `--enable-prefix-caching` or set `...=force` and
  accept the risk.
* **Speculative decoding (MTP/EAGLE).**  Fully supported with a *uniform* layout: all layers
  keep the same number of tokens, and the draft model sees the identical compressed view.
* **Memory.**  The evicted blocks stay allocated to the request until it finishes (a pure
  worker-side monkeypatch cannot hand blocks back to the engine-core scheduler).  The win is
  attention compute and KV-access bandwidth, plus the kvpress quality effect; *block memory is
  not reclaimed in v0.1*.
* **Supported models.**  Plain full-attention (GQA) decoder models — Qwen2/3/3.5, Llama,
  Mistral, ... .  MLA / hybrid (Mamba) KV groups are detected and skipped.
* **Failure behaviour.**  Every hook is wrapped; on any unexpected API mismatch the engine logs
  and continues uncompressed.  Check the log lines starting with `[kvpress-ascend]`.

## Offline simulation (no NPU needed)

The repository contains a full offline simulation that drives the exact engine code against
faithful mocks of vllm-ascend's worker objects and validates the end-to-end invariant
(attention over the compressed layout == attention over the kept tokens):

```bash
python -m pytest tests/ -v
```

## How it is wired (for maintainers)

See `kvpress_ascend/engine.py` — the patched seams are:

* `vllm_ascend.worker.model_runner_v1.NPUModelRunner.execute_model` (prefill tracking +
  compression pass),
* `NPUModelRunner._prepare_inputs` (block-table row rewrite),
* `NPUModelRunner._build_attention_metadata` (seq-lens correction + per-layer overrides),
* `vllm_ascend.worker.block_table.BlockTable.compute_slot_mapping(_draft)` (position shift),
* `vllm_ascend.attention.attention_v1.AscendAttentionBackendImpl.forward` (query capture),
* `vllm.model_executor.layers.attention.Attention.forward` (hidden-state capture for
  layer-importance policies).

All verified against vllm-ascend v0.23.0 / vllm 0.23.0.
