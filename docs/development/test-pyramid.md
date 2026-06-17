# llama.cpp 测试金字塔（推理引擎定制开发）

> 读者对象：任何会动 `src/`、`ggml/src/`、本 fork 新增的自定义后端
>（`ggml/src/ggml-{triton,tilelang,cutlass}/`）、`tools/`、`common/` 或
> `src/models/` 里模型文件的人。本 fork 做端到端校验时通常用转换后的
> **MiniMind-3** GGUF（见 [`minimind-integration.md`](minimind-integration.md)）。

本文是分层测试方案：5 级，从"什么都跑"到"只跑你改过的那一档"。选跟
你刚刚的改动对应的级别——跑高了浪费（5 分钟一次的 Level 1 没人受得了），
跑低了太浅（在 `ggml_mul_mat` 上跑 Level 3 抓不到东西）。

## Level 1 — 无条件门禁

每次 commit / push 前必跑。这是 `CONTRIBUTING.md` 的最低要求。

```bash
cmake --build build --config Release -j$(nproc)
cd build && ctest -L main --verbose --timeout 900
```

能拦住构建断、缺头文件、测试 setup 写挂了这种回归。**拦不住** ggml 算子
的数值回归——那是 Level 3 的活。

在慢 CI runner 上可以跳过昂贵的 `tokenizer` label：

```bash
ctest -L main -E tokenizer --verbose --timeout 900
```

## Level 2 — 用你的模型做端到端冒烟

如果改了**推理路径**上的东西（loader、sampler、KV cache、后端选择、
chat template、grammar、server 任务调度），把模型跑起来，确认它还能
输出像样的文本。

```bash
# 单次补全——练 load → tokenize → decode 循环 → sample → print
./build/bin/llama-cli -m <your>.gguf -p "1+1等于几" -n 50

# 多轮聊天——练 chat template、消息结构、多次 decode
./build/bin/llama-cli -m <your>.gguf -cnv

# HTTP API 路径——练 server 队列、context 调度、批处理
./build/bin/llama-server -m <your>.gguf --port 8080 &
curl -s http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"hi"}]}' | jq .
```

调试图形状（graph shape）时用 `examples/eval-callback/`——它会把图构造器
发出的每个 op 的 tensor 名字 / 形状 / 类型 / 指针都 dump 出来：

```bash
./build/bin/llama-eval-callback -m <your>.gguf -p "hi" -n 5 --no-display-prompt
```

## Level 3 — 按改动范围定向跑

**只跑**和你改的内容对得上的那行。每行就是 1~2 条 `ctest -R`，最慢也不
会超过 30 秒。

| 改了什么 | 该跑的测试 | 为什么是这些 |
|---|---|---|
| **任何 ggml op**（matmul / rope / norm / softmax / silu / quant / dequant / ...） | `test-backend-ops` | 每个 op 都跟参考实现对一遍。**`CONTRIBUTING.md` 强制要求**。 |

> **ggml-triton op 覆盖现状（截至 B.2 / RoPE）：**
> - `GGML_OP_UNARY` (GELU, SILU — fp16 + fp32)
> - `GGML_OP_RMS_NORM` (unweighted + weighted — fp16 + fp32, 4 impls)  *B.1*
> - `GGML_OP_ROPE` (NORMAL + NEOX + MROPE × fp16 + fp32, 6 impls; each dispatches to 4 AOT variants for fwd/bwd × YaRN on/off)  *B.2*
> - `GGML_OP_ROPE_BACK` (由同一组 6 个 ROPE impl 通过 constexpr SIN_SIGN 覆盖)  *B.2*
> - `GGML_OP_ADD` / `GGML_OP_MUL` (TileLang, conditional on `GGML_TRITON_HAS_TILELANG`)
> - `GGML_OP_MUL_MAT` (CUTLASS, conditional on `GGML_TRITON_WITH_CUTLASS`)
>
> 完整 list 见 `scripts/kernel_registry.json` (kernels 数组) 以及
> 每个 provider 的 `register_impl` 调用。
| **量化** | `test-quantize-fns`、`test-quantize-perf`、`test-quantize-stats`、`test-quant-type-selection`、`test-autorelease` | 量化全链路 |
| **RoPE / 位置编码** | `test-rope` | 精准命中——同时也覆盖 YaRN |
| **模型 loader / 架构识别 / tokenizer 接线** | `test-llama-archs`、`test-gguf`、`test-gguf-model-data`、`test-model-load-cancel` | `test-llama-archs` 会为**每一个**已注册架构跑"建图 + 前向 + 写回 GGUF"——loader 最强的端到端测试 |
| **KV cache / 内存** | `test-save-load-state`、`test-state-restore-fragmented`、`test-recurrent-state-rollback`、`test-alloc` | 状态持久化 + 分配器回归 |
| **采样** | `test-sampling`、`test-backend-sampler` | 采样输出的统计校验 |
| **Grammar / 结构化输出** | `test-llama-grammar`、`test-grammar-parser`、`test-grammar-integration`、`test-grammar-llguidance`、`test-gbnf-validator` | 同时覆盖 `<tool_call>` 风格 tool calling |
| **Tokenizer** | `test-tokenizer-0`、`test-tokenizer-1-bpe`、`test-tokenizer-1-spm` | 跳过 `test-tokenizers-repo.sh`（CI 标 `slow`，~1h） |
| **Chat / 模板 / 解析** | `test-chat`、`test-chat-template`、`test-jinja`、`test-chat-peg-parser`、`test-chat-auto-parser`、`test-peg-parser`、`test-json-partial`、`test-regex-partial`、`test-reasoning-budget` | 模板层逻辑全套 |
| **多线程 / 并行 decode** | `test-thread-safety`、`test-barrier` | 并发回归 |
| **计算图优化** | `test-opt`、`test-double-float` | pass 重排 + fp 精度 |
| **GGUF 格式本身** | `test-gguf`、`test-gguf-model-data` | 格式合规 |
| **CLI / arg parser** | `test-arg-parser`、`test-log` | 参数解析 + 日志 |
| **新增了架构**（你加了 `LLM_ARCH_*`） | `test-llama-archs`（自动）+ 该测试在每个架构上跑的图 roundtrip | 同时要更新 `CODEOWNERS`、`HOWTO-add-model.md`、HF→GGUF 转换器 |

跑法：

```bash
cd build && ctest -R '^test-rope$' --verbose                    # 单个
cd build && ctest -R '^test-(rope|backend-ops|llama-archs)$'    # 多个
```

## Level 4 — 跨后端一致性（本 fork 的自定义后端）

本 fork 出了三个新的 ggml 后端——`ggml-triton`（Triton AOT）、
`ggml-tilelang`（TileLang）、`ggml-cutlass`（CUTLASS 3.x）。它们通过
`ggml/src/ggml-backend-reg.cpp` 里的静态注册路径挂上来。

改**任何** ggml op 时，跨后端一致性检查都是**必跑**的：

```bash
./build/bin/test-backend-ops
```

这个测试**只有在编进了 ≥2 个后端时才有意义**。本地推荐矩阵是
**CPU baseline + 三个新后端里的任意一个**。单后端构建只能抓 CPU
侧的精度 bug，抓不到后端之间发散的 bug。

Triton 后端的 AOT 侧，如果动了 `scripts/kernel_registry.json` 或者
`triton_kernels/*.py` 里的源，还要重跑 kernel registry 校验：

```bash
python scripts/compile_kernels.py --out-dir ggml/src/ggml-triton/kernels/generated
cmake --build build --target ggml-triton -j$(nproc)
```

TileLang 后端对应的是 lower-pipeline 驱动；CUTLASS 是重编 `.cu` kernel。
完整构建矩阵见 [`ggml-custom-backends.md`](ggml-custom-backends.md) §7。

## Level 5 — 性能门禁（动 compute 的 PR）

`CONTRIBUTING.md` 说得很死：只要 PR 碰了 compute、KV cache、sampling
或 backend 代码，**必须**在 PR 里附上 perplexity 和性能数据。标准工具链：

```bash
# 量化回归：相对基线波动应该在 ±0.1 以内
./build/bin/llama-perplexity -m <your>.gguf -f wikitext-2.txt

# 吞吐：应该 ≥ 基线 t/s
./build/bin/llama-bench    -m <your>.gguf -p 512 -n 128
```

改之前和改之后**都要**重跑这两个命令并 diff。本 fork 用的模型在同一
机器上的典型基线长这样（数字按你机器填）：

```text
model               size       params  backend    test      t/s
minimind-3 Q8_0     ~64 MiB    64M     CPU        pp512     ~xx
minimind-3 Q8_0     ~64 MiB    64M     CPU        tg128     ~xx
```

（上面数字是占位符，请按自己机器填。）跨机器比绝对 t/s 没意义——只有
**同一台机器**上的 delta 算数。

## Level 0 — 基线检查（新贡献者）

上面任何一级之前，先确认在一个干净的 checkout 上构建本身是过的：

```bash
cmake -B build -DLLAMA_FATAL_WARNINGS=ON -DGGML_RPC=ON
cmake --build build --config Release -j$(nproc)
(cd build && ctest -L main --verbose --timeout 900)
```

这跟 CI 的 `build-cpu.yml` 是同一套。

## 组合使用

**做引擎定制时**的典型内循环：

```bash
# 1. 改了 src/llama-graph.cpp
cmake --build build --config Release -j$(nproc)
ctest -R '^test-(llama-archs|backend-ops|opt)$' --verbose

# 2. 跟基线比 perplexity 和 bench
./build/bin/llama-perplexity -m <your>.gguf -f wikitext-2.txt
./build/bin/llama-bench    -m <your>.gguf -p 512 -n 128

# 3. 端到端冒烟
./build/bin/llama-cli -m <your>.gguf -p "..." -n 50

# 4. 以上都干净了再 commit。
```

**做 kernel / backend 工作**（Triton、CUTLASS 等）时的典型内循环：

```bash
# 1. 改了 ggml/src/ggml-{triton,cutlass,tilelang}/* 或 triton_kernels/*.py
python scripts/compile_kernels.py        # 如果是 Triton
cmake --build build --config Release -j$(nproc)
ctest -R '^test-backend-ops$' --verbose
./build/bin/llama-cli -m <your>.gguf -p "..." -n 50
```

**做 chat / tool-call 工作**时的典型内循环：

```bash
cmake --build build --config Release -j$(nproc)
ctest -R '^test-(chat|chat-template|jinja|chat-peg-parser|chat-auto-parser|grammar|json-partial|regex-partial)$' --verbose
./build/bin/llama-cli -m <your>.gguf -cnv
```

## 相关阅读

- [`minimind-integration.md`](minimind-integration.md) ——先把模型喂进引擎。
- [`ggml-custom-backends.md`](ggml-custom-backends.md) ——新增 ggml 后端
  （CUTLASS、TileLang、Triton-AOT）的架构。
- `CONTRIBUTING.md` §"Coding guidelines" —— 性能门禁的官方措辞。
- `docs/build.md` ——`-DGGML_*` 各个 flag 到底干啥的。
- `tests/test-backend-ops.cpp` ——花 5 分钟通读一遍，知道你"白嫖"了哪些
  覆盖。
