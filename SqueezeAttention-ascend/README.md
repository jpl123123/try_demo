# SqueezeAttention-ascend

Monkeypatch adapter that ports **SqueezeAttention** ([arXiv:2404.04793](https://arxiv.org/abs/2404.04793),
Wang & Gan, 2024) — *2D management of KV-cache: layer-wise optimal budget allocation* — to
**vllm-ascend** (vLLM v1, Ascend NPU), without touching the installed `vllm` / `vllm-ascend`
sources.

SqueezeAttention measures each decoder layer's importance during prefill (cosine similarity
between the layer's input and output hidden states), clusters the layers with KMeans into three
classes and gives each class its own KV budget.  Every layer then keeps its budgeted KV pairs in
a streaming-LLM fashion (sink tokens + most recent tokens).  The original project ships as
`modeling_llama.py` / `modeling_mistral.py` replacements for transformers; this package runs the
same algorithm inside the vLLM v1 worker via the shared [kvpress-ascend](../kvpress-ascend)
patch engine.

## Quick start (on the NPU machine)

```bash
# 1. install (kvpress-ascend is pulled in automatically)
pip install ./SqueezeAttention-ascend

# 2. launch vllm-ascend as usual, with one extra export:
export squeeze=1          # activates the patch (also: SQUEEZE, SQUEEZE_ASCEND, ...)
vllm serve /softwarePlatform/c00879303/Qwen3.5-27B-w8a8-mtp ...   # your normal command
```

Like kvpress-ascend, this package installs a `squeezeattention_ascend.pth` so every Python
process (API server, engine core, TP workers) imports it at startup; without the export it is
inert.

## How it works on vLLM

1. The engine captures, for every prefill chunk, the per-layer cosine similarity between each
   attention module's input and output hidden states (running mean).
2. At prefill completion the SqueezeAttention press runs KMeans(3) over the per-layer means and
   allocates budgets: the most important class gets `class3_ratio × prompt_len` tokens per
   layer, the others `a × prompt_len`, with `a` chosen so the **total budget is preserved**
   (`num_layers × ini_size × prompt_len`).
3. The engine compacts each layer's paged cache to its own budget (sink + recent tokens per
   layer) and rewrites the per-layer block tables / slot mappings / seq-lens so the attention
   kernels only see the kept tokens.
4. **With speculative decoding (MTP/EAGLE)** the layout automatically collapses to a uniform
   budget (the mean of the per-layer budgets): the draft model shares the group-0 cache and
   needs one block layout.  The total budget is still preserved.

## Environment variables

| Variable | Default | Meaning |
|---|---|---|
| `squeeze` / `SQUEEZE` / `SQUEEZE_ASCEND` / `SQUEEZE_ASCEND_ENABLED` | – | activate the patch |
| `SQUEEZE_ASCEND_INI_SIZE` | `0.21` | average budget per layer (fraction of prompt length) |
| `SQUEEZE_ASCEND_CLASS3_RATIO` | `0.08` | budget of the most important layer class |
| `SQUEEZE_ASCEND_SINK` | `4` | streaming-LLM sink tokens |
| `SQUEEZE_ASCEND_KMEANS_SEED` | `0` | KMeans seed |
| `SQUEEZE_ASCEND_PER_LAYER` | `1` | per-layer layouts (auto-disabled under spec decode) |
| `SQUEEZE_ASCEND_LOG` | `info` | log level |

## Notes

* Same caveats as kvpress-ascend: prefix caching (`KVPRESS_ASCEND_PREFIX_CACHE=skip` default),
  block memory is not reclaimed in v0.1, plain full-attention models only.
* If both `export kvpress=1` and `export squeeze=1` are set, one policy must win: set
  `KVPRESS_ASCEND_POLICY=squeeze` to prefer SqueezeAttention (default is `kvpress`).
* Offline simulation (no NPU): `python -m pytest tests/ -v`.
