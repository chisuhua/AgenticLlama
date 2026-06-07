# 将 MiniMind-3 模型集成进 llama.cpp

> 读者对象：本地已有 Hugging Face 模型（或任何非 llama.cpp 自带权重）的开发者，
> 既要把它跑通推理，又要在推理引擎上做二次开发。

本文是 **MiniMind-3**（jingyaogong 的教育型 64M Dense / 198M-A64M MoE 模型，
架构对齐 Qwen3，词表是自训的 6400 大小）接入 llama.cpp 推理引擎的规范流程，
以及途中会踩到的坑。大部分步骤对任何"HF 格式 + 自定义 tokenizer"的小模型
都通用。

## 1. 示例里到底带不带模型？

**不带。** `examples/` 下的推理示例（`examples/simple`、`examples/simple-chat`、
`examples/batched`、`examples/parallel` 等）一个都不附权重。它们的 README
里引用的具体模型（如 `llama-7b-v2`、`Meta-Llama-3.1-8B-Instruct`）只是举例，
运行时统一由 `-m <path-to-gguf>` 指定。

引擎真正接受的格式只有 **GGUF**。其它格式（Hugging Face safetensors、
PyTorch `.pth`、老 GGML）都要先转换。

## 2. MiniMind-3 架构速览表

动手前先读 `minimind-3/config.json`——下表就是引擎实际要消费的东西：

| 字段 | 取值 | 引擎侧的含义 |
|---|---|---|
| `architectures` | `["Qwen3ForCausalLM"]` | 上游已支持——不需要新加 `LLM_ARCH_*` |
| `model_type` | `qwen3` | 走 `src/models/` 里的 Qwen3 图构造器 |
| `hidden_size` | 768 | 很小，迭代快，GGUF < 200MB |
| `num_hidden_layers` | 8 | 极少，测试里完整 decode 一遍也没压力 |
| `num_attention_heads` | 8 | 标准 MHA 路径 |
| `num_key_value_heads` | 4 | GQA——顺带把 grouped-QKV 路径也走一遍 |
| `head_dim` | 96 | 768 / 8 |
| `intermediate_size` | 2432 | SwiGLU FFN 维度 |
| `hidden_act` | `silu` | SwiGLU——silu×gate matmul 配对 |
| `vocab_size` | 6400 | **自定义**——与 Qwen3 的 151643 不匹配 |
| `max_position_embeddings` | 32768 | RoPE 位置 |
| `rope_theta` | 1e6 | Qwen3 风格 RoPE base |
| `rope_scaling` | `null` | 推理时通过 `--inference_rope_scaling` 启用 YaRN |
| `tie_word_embeddings` | `true` | embedding 与 output 共享权重，Qwen3 路径已处理 |
| `dtype` | `float16` | 引擎默认按 FP16 反量化 |

唯一真正需要动手术的是 **tokenizer**。MiniMind 用了自己训的 6400 词表
BPE+ByteLevel tokenizer，上游 `convert_hf_to_gguf.py` 不认识；只有这一处
需要人工干预。

## 3. 转换：HF → GGUF

### 3.1 给 tokenizer 加个兜底分支

在 `convert_hf_to_gguf.py` 的 `get_vocab_base_pre()` 函数末尾加一段：

```python
# ---- MiniMind（以及类似的 6400 词表自定义 BPE）----
# MiniMind-3 的 HF tokenizer 不是已知模型，但 tensor 布局是 Qwen3，
# 所以临时复用 qwen2 的 tokenizer 处理路径。如果你的模型更接近别家，
# 把下面这个 "qwen2" 换成行为最接近的那个。
if res is None:
    res = "qwen2"
```

这是**临时**的兜底——正经做法是在 `gguf-py/gguf/constants.py` 里加一个
`MODEL_ARCH.MINIMIND` 枚举，再写一个 `ModelBase.register("MiniMindForCausalLM")`
子类。但对开发迭代来说这个兜底完全够用，upstream MiniMind 的 README 也是
这么建议的。

### 3.2 执行转换

```bash
# 在 llama.cpp 仓库根目录跑
python convert_hf_to_gguf.py ./minimind-3/
# 产物：./minimind-3/<something>-F16.gguf
```

### 3.3 （可选）量化

```bash
./build/bin/llama-quantize \
    ./minimind-3/xxx-F16.gguf \
    ./minimind-3/xxx-Q8_0.gguf Q8_0
```

开发迭代时建议**同时留一份 FP16**——量化会引入噪声，perplexity 数字会
更难看，diff 起来反而不直观。

### 3.4 冒烟测试

```bash
./build/bin/llama-cli -m ./minimind-3/xxx-F16.gguf -p "你好,介绍下你自己" -n 200
./build/bin/llama-cli -m ./minimind-3/xxx-F16.gguf -cnv              # 聊天模式
./build/bin/llama-server -m ./minimind-3/xxx-F16.gguf --port 8080    # HTTP API
```

`minimind.modelfile` 自带 ChatML 风格模板（`<|im_start|>` / `<|im_end|>`）和
`<think>` 思考标签。`llama-cli` 会从 GGUF 内嵌的 chat template 里自动读
出来；只有在外部单独保存模板时才需要 `--chat-template`。

## 4. 保留为「基线模型」

拿到一份干净的 FP16 GGUF 之后：

1. 在一个干净的 `master` commit 上跑一次 `llama-perplexity` 和
   `llama-bench`，把数字记下来。这就是你的**基线**。
2. 把这份 GGUF 放到**本地**某个非跟踪路径（`/tmp/minimind-3-F16.gguf` 之类）。
   仓库的 `.gitignore` 对此友好——**不要**把它 commit 进被跟踪的树。
3. 每次改了 compute / KV cache / sampling / backend 代码后，再跑这两个
   命令，跟基线做 diff。`CONTRIBUTING.md` 明确规定这就是受影响的 PR
   必须过的**性能门禁**。

完整的测试规划见 [`test-pyramid.md`](test-pyramid.md)。

## 5. 常见坑（先看这一节）

- **`Unknown model architecture: MiniMindForCausalLM`** ——忘了打 tokenizer
  补丁。`convert_hf_to_gguf.py` 在跑 tensor 映射**之前**就读 `config.json`
  里的 architecture 行。先把 §3.1 的 patch 加上。
- **输出全是像随机 token 的乱码** —— RoPE theta 错了，或 YaRN scaling 没
  打开。MiniMind-3 用的是 `theta=1e6`；如果你在引擎日志里看到
  `rope_freqs.weight` 相关的告警，回头检查 `src/llama-model.cpp` 的
  `llama_model_rope_type()` 里 Qwen3 分支。
- **`vocab size mismatch` 警告** —— 模型被路由到了错误的架构。确认
  `llama_model::build_graph()` 里走的是 Qwen3 分支。
- **decode 出来满屏 `<unk>`** —— GGUF 里的 `tokenizer.ggml` 表没带上 BPE
  merges。回去确认 §3.1 的 patch 确实触发了（不放心就 `print(res)` 一下）。

## 6. 相关阅读

- [`test-pyramid.md`](test-pyramid.md) ——在转换后的模型上做引擎开发时
  使用的 5 级测试金字塔。
- [`../development/ggml-custom-backends.md`](../development/ggml-custom-backends.md) ——
  本 fork 新增 ggml 后端（CUTLASS、TileLang、Triton-AOT）的架构。
- [`../HOWTO-add-model.md`](../development/HOWTO-add-model.md) ——添加全新
  架构的完整流程（MiniMind-3 不需要——Qwen3 已被支持——但如果你 fork 出
  变体可能用得上）。
- 上游 MiniMind README §"llama.cpp 集成"——同款流程，只是用上游中文表达。
