# vllm-ascend KV 压缩工具套件

把 **kvpress** 与 **SqueezeAttention** 两个 KV-cache 压缩方法，以**纯 monkeypatch** 方式适配到
**vllm-ascend**（vLLM v1 引擎、Ascend NPU）——不改动 `vllm` / `vllm-ascend` 任何源码。

本仓库包含两个可独立 `pip install` 的包：

| 目录 | 包 | 说明 |
|---|---|---|
| [`kvpress-ascend/`](kvpress-ascend/) | `kvpress-ascend` | NVIDIA kvpress 的 vllm-ascend 适配：SnapKV / StreamingLLM / Random / Knorm / TOVA / PyramidKV / AdaKV 等压测策略 + 共享压缩引擎（块表重写、槽位移、seq-lens 修正） |
| [`SqueezeAttention-ascend/`](SqueezeAttention-ascend/) | `SqueezeAttention-ascend` | SqueezeAttention 逐层预算分配（KMeans 分层 + 流式保留），运行在 kvpress-ascend 的引擎之上 |

## 快速开始（NPU 机器上）

```bash
pip install ./kvpress-ascend ./SqueezeAttention-ascend

export kvpress=1        # 启用 kvpress 压缩（默认 SnapKV，ratio 0.5）
# 或
export squeeze=1        # 启用 SqueezeAttention（KVPRESS_ASCEND_POLICY=squeeze 可覆盖默认）
# 或
export KVPRESS_ASCEND_DRY_RUN=1   # 只算分不写缓存，先观察

vllm serve ...          # 你原有的 vllm-ascend 拉起命令，原样不动
```

两个包都通过 `.pth` 自动激活（解释器启动即导入，env 未设置时完全惰性），详细用法、
环境变量与已知限制见各包 README。

## 无硬件离线验证

两个包都自带**离线模拟测试**（无需 NPU / 无需 vllm）：

```bash
python -m pytest kvpress-ascend/tests SqueezeAttention-ascend/tests -v
python -m kvpress_ascend.simulate      # 自检 CLI
```

端到端不变量：压缩布局上的注意力输出 == 参考保留 token 集合上的注意力输出（误差 < 1e-4）。

## 已知限制（v0.1，详见各包 README）

- 与 `--enable-prefix-caching` 冲突：默认 `KVPRESS_ASCEND_PREFIX_CACHE=skip`（安全跳过），
  需压缩时去掉该参数或 `force` 自担风险；
- 纯 worker 侧 patch 无法回收块内存（省的是注意力计算/带宽）；
- MTP/EAGLE 投机解码下自动使用统一布局（draft 与 target 同一压缩视图）；
- 仅支持纯全注意力（GQA）模型组，MLA/混合组自动跳过。

## License

Apache-2.0（见 [LICENSE](LICENSE)）。
