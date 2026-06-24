# llama.cpp fork — 项目目标与文档入口

> 本 fork 的**单一目标**：
> **把 MiniMind-3（jingyaogong 训的 64M/198M 教育型小模型）端到端跑通
> llama.cpp 推理引擎，同时用自研的统一 Triton 多 kernel 后端
> (CUTLASS + TileLang + Triton-AOT) 来加速 GPU 推理。**
>
> 一切架构决策、文档、PR 都以这个目标为锚。偏离它的"通用可插拔后端框架"、
> "对标上游 PR"之类的动机都放在第二位。

本文是 `docs/development/` 的**入口文档**——从这里出发，按下面的链接找到
对应子任务的详细文档。所有详细页都假设你已经读过本页。

## 当前状态

| 子目标 | 状态 | 主要 PR / 文档 |
|---|---|---|
| **1. 统一 Triton 多 kernel 后端**（CUTLASS + TileLang 折进 ggml-triton） | ✅ 完成（PR #1，squash-merged into master） | [`PR #1`](https://github.com/chisuhua/AgenticLlama/pull/1)、计划 `.omo/plans/2026-06-08-unify-triton-multi-kernel-backend.md`、`docs/performance/unified-backend.md` |
| **2. MiniMind-3 端到端集成**（HF → GGUF → llama-cli/llama-server） | ✅ 文档完成 + 基线（CPU-only）已捕获 | [`minimind-integration.md`](minimind-integration.md) |
| **3. 端到端 GPU 验证 + 性能门禁**（4 个 mode 的 perplexity + llama-bench） | ⏸ 部分 deferred 到 GPU host | `docs/superpowers/plans/baseline-deferral.md` "What to do on a GPU host" 一节 |
| **4. kernel 集扩展**（逐步覆盖 Qwen3 推理图上所有 op） | ✅ **B 阶段完成** (B.1 RMSNorm + B.2 RoPE + B.3 FlashAttn 全部 22 个 Triton AOT impls；`test-triton-registry` 6 asserts 全过；CPU-only host 验证；**数值验证 deferred 到 GPU host**) | `ggml/src/ggml-triton/ggml-triton-provider-{rmsnorm,rope,flash-attn}.{h,cpp}`、3 份 plan `docs/superpowers/plans/2026-06-{10,12,21}-*-triton-aot.md`、ROADMAP §1 / §3 |

如果只读一份文档就读这个表——剩下的都是展开。

> **下一步要做什么？** 完整答案见 [`ROADMAP.md`](ROADMAP.md)——4 阶段 (A/B/C/D)、
> 依赖图、工作量估算、风险条款。

## 阅读顺序（按目标推进路径）

```
[0] 本页（项目目标 + 文档地图）  ←  现在
 │
 ▼
[1] docs/development/ggml-custom-backends.md
    架构总览：Provider 抽象、kernel registry、dispatch 逻辑、
    CUTLASS/TileLang/Triton 各自定位。
    不读这个改 kernel 就是在盲改。
 │
 ▼
[2] docs/backend/TRITON.md
    用户视角的 Triton 后端：构建、CMake 选项、模型准备、运行。
    "我要跑一个模型过 Triton 后端"——按这个走。
 │
 ▼
[3] docs/backend/CUTLASS-TO-TRITON-MIGRATION.md
    docs/backend/TILELANG-TO-TRITON-MIGRATION.md
    PR #1 之前用过 standalone 后端的用户必读——旧选项会发 deprecation
    warning，迁移步骤在这里。
 │
 ▼
[4] docs/development/test-pyramid.md
    5 级测试金字塔：动 ggml op / provider / dispatch 表时跑哪一级。
    每次 PR 之前至少过 Level 1。
 │
 ▼
[5] docs/development/minimind-integration.md
    端到端：HF → GGUF → llama-cli 冒烟 → 跑 perplexity / llama-bench
    与基线 diff。这就是 PR #1 之后这个 fork 的"目的"——验证统一后端
    没有打破 MiniMind-3 的推理。
 │
 ▼
[6] docs/superpowers/plans/baseline-deferral.md
    性能门禁中 GPU 验证步骤的精确 recipe（perplexity 命令、llama-bench
    命令、4 个 mode 的对比表）。在 GPU host 上跑这一份补全 §3。
```

## 任务 1：Triton 后端（架构 + 迁移 + 性能）

本 fork 在 `ggml/src/ggml-triton/` 下做了**一个**后端，里面挂**多个**
kernel provider：

| Provider | 用途 | 入口 |
|---|---|---|
| `cpu` | 参考实现 + 无 GPU 时的 fallback | `ggml-triton-provider-cpu.cpp` |
| `triton` | OpenAI Triton DSL → AOT → CUBIN | `ggml-triton-provider-triton.cpp` + `scripts/compile_kernels.py` |
| `cutlass` | NVIDIA CUTLASS 3.x C++ 模板（GEMM） | `ggml-triton-provider-cutlass.cpp` + `kernels/cutlass/*.cu` |
| `tilelang` | TileLang DSL → TVM lower → CUDA source（elementwise） | `ggml-triton-provider-tilelang.cpp` + `kernels/tilelang/generated/*.cu` |

构建/迁移/性能对照，按下面的顺序读：

1. **架构**（必读）：[`ggml-custom-backends.md`](ggml-custom-backends.md) —
   Provider 抽象、kernel registry 协议、动态加载、4 个 provider 各自的
   状态机、为什么折进同一个后端而不是各立一摊。
2. **用户视角**（构建/跑模型）：[`../backend/TRITON.md`](../backend/TRITON.md) —
   CMake 选项矩阵、操作系统/硬件支持表、运行验证。
3. **从老后端迁移过来**（如果用过 PR #1 之前的 standalone `GGML_CUTLASS`
   或 `GGML_TILELANG`）：
   - [`../backend/CUTLASS-TO-TRITON-MIGRATION.md`](../backend/CUTLASS-TO-TRITON-MIGRATION.md)
   - [`../backend/TILELANG-TO-TRITON-MIGRATION.md`](../backend/TILELANG-TO-TRITON-MIGRATION.md)
4. **性能门禁的对比基线**（PR #1 之后填）：[`../performance/unified-backend.md`](../performance/unified-backend.md) —
   4 个 mode（cutlass-standalone / cutlass-via-triton / tilelang-standalone /
   tilelang-via-triton）的 perplexity + t/s 矩阵。**正确性等价性** 是
   PR #1 的设计核心——kernels 字节完全一致、launcher C ABI 不变、op 粒度
   分派，理论上 4 个 mode 数字必须完全一致；实际数字等 GPU host 上跑出来
   回填到该文档。

PR #1 的执行计划留作历史档案：
`.omo/plans/2026-06-08-unify-triton-multi-kernel-backend.md`。
Momus 已审过，5 个阶段全部完成。

## 任务 2：MiniMind-3 端到端集成

**这一节是 fork 的"目的"本身**——前面那个 Triton 后端是手段，跑通
MiniMind-3 才是目标。

详细 recipe：[`minimind-integration.md`](minimind-integration.md)。

简版流程（细节全在上面的链接里）：

```
HF model (./minimind-3/)
    │
    │  python convert_hf_to_gguf.py ./minimind-3/
    │   (需要 §3.1 的 tokenizer 兜底 patch)
    ▼
FP16 GGUF
    │
    │  ./build/bin/llama-cli -m model-F16.gguf -p "你好" -n 200
    │  (冒烟测试：能跑出中文就是通了)
    ▼
成功跑出 MiniMind-3 的中文输出
    │
    │  ./build/bin/llama-perplexity -m model-F16.gguf -f wiki.test.raw
    │  ./build/bin/llama-bench -m model-F16.gguf
    │  (这是基线，每次 PR 都要跟它 diff)
    ▼
端到端集成 + 性能门禁建立
```

MiniMind-3 选作参考模型不是偶然——它有 4 个**端到端校验最看重**的特性：
- 极小（64M / 198M）→ ctest 5 分钟能跑完
- 架构对齐 Qwen3 → 上游已有 `LLM_ARCH_QWEN3`，不需要新加模型代码
- 自训 6400 词表 → 逼着 `convert_hf_to_gguf.py` 走 tokenizer 兜底分支
  （这是上游 MiniMind README 推荐的入口——也是把任何"HF 格式 + 自定义
  tokenizer"的小模型接进 llama.cpp 的标准流程）
- 配套 `<think>` + ChatML 模板 → 顺便把 chat / 推理 / 思考标签路径走一遍

## 跨任务关注点

- **测试规划**：[`test-pyramid.md`](test-pyramid.md) — 5 级金字塔。
  动 kernel 源/provider/dispatch 表/模型代码时跑哪一级、对应命令、CI
  矩阵全在这里。**改 PR 之前先选档**。
- **基线 / 复现**：[`../superpowers/plans/baseline-deferral.md`](../superpowers/plans/baseline-deferral.md) —
  CPU-only 基线 + GPU 验证步骤的精确 recipe。在 GPU host 上回填 §3 之前，
  CI runner 上跑 §0.1 的 CPU 冒烟是最低门槛。
- **性能文档**：[`../performance/unified-backend.md`](../performance/unified-backend.md) —
  4-mode 对比表的填空区。GPU host 上跑完之后回填。

## 仓库里的相关位置一览

```
ggml/src/ggml-triton/                   # 统一后端源码
├── ggml-triton.cpp                     #   注册入口、device/reg 协议
├── ggml-triton-provider.cpp            #   provider 抽象、registry
├── ggml-triton-provider-cpu.cpp        #   CPU 参考实现
├── ggml-triton-provider-cutlass.cpp    #   CUTLASS GEMM provider
├── ggml-triton-provider-tilelang.cpp   #   TileLang elementwise provider
├── ggml-triton-provider-triton.cpp     #   Triton AOT provider
├── ggml-triton-dispatch.cpp            #   调度逻辑
└── kernels/
    ├── cutlass/                        #   CUTLASS GEMM .cu 源
    ├── tilelang/generated/             #   TileLang 预生成 .cu
    └── generated/                      #   Triton AOT CUBIN (embedded C)

scripts/                                # AOT 驱动 + kernel registry
├── compile_kernels.py
├── kernel_registry.json
└── bench-backend.sh                    #   4-mode perplexity + llama-bench 驱动

tests/test-triton-registry.cpp          # provider 注册的单元测试 (PR #1 新增)

.omo/plans/                             # 历史执行计划
└── 2026-06-08-unify-triton-multi-kernel-backend.md

minimind-3/                             # 端到端测试用的参考模型（不跟踪，仅供本地）
```

## 何时更新本页

- ✅ 一个子任务状态从"⏸ deferred"变成"✅ 完成"时——更新本文状态表
- ✅ 添加新的子目标时——在状态表加一行、加一节"任务 N"
- ✅ 子任务的入口文档有变动时——更新本文的链接

## 相关阅读（按距离本目标的远近）

最近（fork 核心）：
- [`ggml-custom-backends.md`](ggml-custom-backends.md) — Triton/TileLang/CUTLASS 架构
- [`minimind-integration.md`](minimind-integration.md) — 端到端 recipe
- [`test-pyramid.md`](test-pyramid.md) — 5 级测试
- [`ROADMAP.md`](ROADMAP.md) — 实施路线图 (4 阶段、依赖图、工作量、风险)

次近（fork 周边）：
- [`../backend/TRITON.md`](../backend/TRITON.md) — 用户视角
- [`../backend/CUTLASS-TO-TRITON-MIGRATION.md`](../backend/CUTLASS-TO-TRITON-MIGRATION.md) — 迁移
- [`../backend/TILELANG-TO-TRITON-MIGRATION.md`](../backend/TILELANG-TO-TRITON-MIGRATION.md) — 迁移
- [`../performance/unified-backend.md`](../performance/unified-backend.md) — 性能
- [`../superpowers/plans/baseline-deferral.md`](../superpowers/plans/baseline-deferral.md) — 基线

远（上游/通用）：
- [`HOWTO-add-model.md`](HOWTO-add-model.md) — 加新模型架构
- [`parsing.md`](parsing.md) — PEG parser
- [`debugging-tests.md`](debugging-tests.md) — 调测测试
- [`token_generation_performance_tips.md`](token_generation_performance_tips.md) — token 性能
- [`ggml-custom-backends.md`](ggml-custom-backends.md) §"已知的失败模式"（待补）
