# 实施路线图 — 从 PR #1 到项目目标

> 读者对象：自己 / 任何接手这个 fork 的人。
>
> 本文是 [`README.md`](README.md) 状态表"如何变成全部 ✅"的具体答案。
> 每条线都锚定到一个可验证的产物上——不是"加强后端"这种空话。

## 1. 现状（截至当前 commit）

- ✅ **Task 1**：Triton 多 kernel 后端统一（PR #1 已 merge，post-merge fix 已 push）
- ✅ **Task 2**：MiniMind-3 端到端文档 + CPU-only 基线（`minimind-integration.md`，perplexity + bench 命令齐全）
- ⏸ **Task 3**：GPU 性能门禁（4-mode 对比表，deferred 到 GPU host，理论正确性已论证）
- 🔄 **Task 4**：kernel 集扩展——**B.1 RMSNorm ✅**（commit `418f88b9f`，unweighted + weighted × fp16/fp32 = 4 impls；`test-triton-registry` 验证；B.2 RoPE / B.3 FlashAttn 待做）

fork 当前**目的性正确性**已立（CPU-only 跑得通 MiniMind-3、4-mode 行为等价、RMSNorm provider 通路打通），
但**目的性性能**还没拿到——需要在 GPU 上把 MiniMind-3 跑在 ggml-triton 后端
上（而不是 fallback 到 ggml-cpu），证明自研后端有实际加速。

## 2. 路线图（按目标推进的顺序）

```
┌─────────────────────────────────────────────────────────────────────────┐
│ Phase A: GPU 端到端验证 (Task 3)        — GPU host                     │
│   A.1 拿一个 GPU host (含真实 NVIDIA 设备)                              │
│   A.2 在 4 个 mode 上跑 MiniMind-3 + llama-perplexity + llama-bench     │
│   A.3 把数字填进 docs/performance/unified-backend.md                  │
│   A.4 修任何在 GPU 跑出来的差异 (理论上不会有)                          │
│                                                                          │
│  ──── 验证: docs/performance/unified-backend.md 的 4-mode 数字全到位 ── │
│                                                                          │
│ Phase B: kernel 集覆盖 Qwen3 (Task 4)   — CPU-only box 可启动           │
│   B.1 RMSNorm provider (fp16/fp32)  — Triton AOT  ✅ (commit 418f88b9f) │
│   B.2 RoPE provider (neox 风格, Qwen3 用)  — Triton AOT                 │
│   B.3 FlashAttn provider (标准 fp16, 不带 KV cache 长上下文)            │
│   B.4 Provider registry 接入 + test-backend-ops 全绿                    │
│                                                                          │
│  ─── 验证: test-backend-ops 上 Qwen3 graph 100% 走 ggml-triton ──────  │
│                                                                          │
│ Phase C: MiniMind-3 on Triton (Task 2 完成) — 需 A + B                  │
│   C.1 -ngl 999 跑 MiniMind-3, 验证 op 路由日志显示 Triton 命中         │
│   C.2 perplexity diff vs CPU-only 基线 ≤ 0.01 (噪声容差)                │
│   C.3 llama-bench 取 4 档 (n_gpu_layers=0/4/8/all) t/s 对比          │
│   C.4 docs/performance/unified-backend.md 增 §"Qwen3/MiniMind-3 实跑" │
│                                                                          │
│  ─── 验证: MiniMind-3 的 pp512 + tg128 数字在 Triton 后端上 ≥ ggml-cpu ─ │
│                                                                          │
│ Phase D: 生产化硬化 (gated on C)        — 任意 host                     │
│   D.1 add SM 70/75/86/89 变体到 kernel_registry.json                    │
│   D.2 unit test 覆盖 provider fallback 路径 (CPU vs GPU 选择)          │
│   D.3 docs/development/ggml-custom-backends.md 补"已知失败模式"节       │
│   D.4 (远期) ROCm 适配评估                                              │
│                                                                          │
│  ─── 验证: -DGGML_TRITON_WITH_CUTLASS=OFF / ON / TILELANG=ON 矩阵全绿 ──│
└─────────────────────────────────────────────────────────────────────────┘
```

## 3. 详细分解

### Phase A — GPU 端到端验证

**目标**：把 4-mode perplexity + t/s 数字落到 `docs/performance/unified-backend.md`。

| 步骤 | 文件/产物 | 验证 |
|---|---|---|
| A.1 | 无代码改动 | `nvidia-smi` 有输出；`/dev/nvidia0` 存在；nvcc 可用 |
| A.2 | `scripts/bench-backend.sh <model.gguf> cutlass-standalone` × 4 | 4 次运行无错 |
| A.3 | `docs/performance/unified-backend.md` 表格填空 | 4-mode 数字都填齐 |
| A.4 | （若有）`ggml-triton-provider-*.cpp` 补丁 + commit | `test-backend-ops` 仍 100% pass |

**依赖**：
- 物理 GPU（不是 CUDA 兼容模式）
- 已转换的 MiniMind-3 GGUF（FP16，~200MB）
- 一个 512-token 长的 perplexity 文本（`wiki.test.raw` 之类）

**不需要**：kernel 改动。Phase A 是**回归性验证**——确认 PR #1 没改 ggml 数值
行为。如果 A.4 出现差异，说明 PR #1 有隐藏 bug，需要回到 `ggml-triton-provider-*.cpp` 排查。

**退出标准**：
- 4-mode perplexity 数字两两之差 ≤ 1e-3（理论应该完全相同，留余量给 FP16 噪声）
- 4-mode t/s 数字差异在 ±5% 以内（launcher C ABI 一致时，launch overhead 一致）
- `test-backend-ops` 全绿
- `docs/performance/unified-backend.md` §"GPU 实测数字"表填齐

### Phase B — kernel 集覆盖 Qwen3

**目标**：让 ggml-triton 跑得动 Qwen3 (MiniMind-3 的架构) 推理图的全部 op。

**Op 覆盖现状 vs 缺口**：

| Op 族 | ggml-triton 内现状 | MiniMind-3 (Qwen3) 是否需要 | 阻塞性 |
|---|---|---|---|
| `MUL_MAT` (GEMM) | CUTLASS, f16/f32/q4_0/q8_0 | ✅ 全 transformer 块 + LM head | — |
| `RMS_NORM` | ✅ Triton AOT (B.1, unweighted + weighted × fp16/fp32 = 4 impls) | ✅ 每层 2 次 (pre-attn + pre-MLP) | 已解决 (B.1 ✅)。Stage 1 限制 `ne00 ≤ 1024` 待 multi-block 变体 |
| `ROPE` (neox) | ❌ 无 | ✅ 每层 2 次 (Q, K) | **高** (B.2 待做) |
| `FLASH_ATTN_EXT` | ❌ 无 | ✅ 每层 1 次 (Q × K^T → softmax → × V) | **高** (B.3 待做，最大块) |
| `GELU` / `SILU` | Triton AOT (f16/fp32) | ✅ SwiGLU gate 用 SiLU | — |
| `ADD` / `MUL` | TileLang (f16/fp32) | ✅ residual add | — |
| `MUL_MAT_ID` (专家) | ❌ 无 | ❌ MiniMind-3 是 Dense 不需要；198M-A64M MoE 才需要 | 低 |

**每个新增 provider 的标准流程**（4 步循环）：

1. **Kernel 源** — 在 `triton_kernels/<op>/` 写 Triton DSL（参考现成
   `gelu/` / `silu/` 的目录结构），在 `scripts/kernel_registry.json` 注册。
2. **AOT 编译产物** — `python scripts/compile_kernels.py --out-dir ggml/src/ggml-triton/kernels/generated/`
   产出 `<op>_<dtype>_sm80.{c,h}`，check-in。
3. **Provider 实现** — `ggml-triton-provider-<op>.{h,cpp}`，实现
   `ggml_triton_provider_i` 接口（参考 `ggml-triton-provider-triton.cpp`），
   在 `ggml_triton_global_registry()` 注册。
4. **测试** — `test-backend-ops` 自动覆盖（每个 ggml op 都有 reference 对比），
   `test-triton-registry.cpp` 加一行 provider 注册断言。

**B.1 RMSNorm 的具体设计**：
- 入口 op：`GGML_OP_RMS_NORM`
- dtype：fp16, fp32（MiniMind-3 用 fp16）
- kernel 算法：标准 `y = x * rsqrt(mean(x^2) + eps) * weight`
- Triton DSL 行数：~30 行（含 `tl.load` / `tl.sum` / `tl.rsqrt` / `tl.store`）
- 复用现有 `elementwise` AOT 框架

**B.1 完成情况**（commit `418f88b9f`，plan 在 `docs/superpowers/plans/2026-06-10-rmsnorm-triton-aot.md`）：
- ✅ 实现按 **unweighted / weighted × fp16 / fp32 = 4 个 impls**——unweighted 是
  MiniMind-3 / `test_rms_norm` 实际走的路径（`src[1]==nullptr`），weighted 走融合
  RMSNorm（`src[1]!=nullptr`）。
- ✅ `test-triton-registry` 加 Assert 4 验证 4 个 triton AOT impls 全部在
  global registry（exit 0）。
- ✅ `test-backend-ops` 在 CPU baseline 路径上仍 100% pass。
- ✅ `GGML_TRITON_WITH_RMSNORM` CMake option（默认 ON）镜像 CUTLASS / TileLang
  模式——CI 可独立 flip 旗标做 bisect。
- ⚠️ **Stage 1 已知限制**（详见 `ggml-custom-backends.md` §"已知失败模式"）：
  - 行长 `ne00 > 1024` → CPU fallback（待 multi-block 变体）
  - eps 烘焙 `1e-6`（`± 1e-7` tolerance 闸门）→ 偏离时 CPU fallback（待 Stage 2 运行时 eps）
  - 真实 CUBIN 生成在 CPU-only host 上是 16-byte ELF-magic placeholder（Phase 0
    audit §0.4，GPU host 上回归后会得到真实 CUBIN）
- 📦 **完整 plan + 14 步 TDD 执行档案**：`docs/superpowers/plans/2026-06-10-rmsnorm-triton-aot.md`。
  关键设计决策（per Oracle review）已写进 plan §1-§10 自审。

**B.2 RoPE 的具体设计**：
- 入口 op：`GGML_OP_ROPE`（neox 风格，Qwen3 用）
- dtype：fp16
- kernel 算法：标准 rotary embedding 复数乘法
- 复用现有 `elementwise` AOT 框架

**B.3 FlashAttn 的具体设计**：
- 入口 op：`GGML_OP_FLASH_ATTN_EXT`
- dtype：fp16
- kernel 算法：标准 FlashAttention-2 (在线 softmax + 块化)
- Triton DSL 行数：~150 行（参考 triton/tutorials/06-fused-attention.py）
- 这是 B 阶段最大的工作量，但**收益也最大**——attention 是 LLM 推理的热点

**退出标准**：
- `test-backend-ops` 100% pass（其中 RMS_NORM / ROPE / FLASH_ATTN_EXT 三个新 op
  在 Triton provider 上跑得通且数值对齐 reference）
- `test-triton-registry.cpp` 3 个新 provider 注册断言通过
- 在 GGML_LOG_LEVEL=DEBUG 下，Qwen3 graph 全部 op 都显示 `ggml-triton` 命中，
  没有 `ggml-cpu` fallback（除了 op reshape/view 这种元数据 op）

### Phase C — MiniMind-3 实跑在 Triton 后端

**目标**：证明 B 阶段补的 kernel 真的有用——MiniMind-3 跑在 ggml-triton 上
比纯 CPU 快。

| 步骤 | 文件/产物 | 验证 |
|---|---|---|
| C.1 | (无新文件) 跑 `./llama-cli -ngl 999 -m model.gguf`，GGML_LOG_LEVEL=DEBUG | 路由日志显示 Triton 命中 QKV/MatMul/RMS/RoPE/Attn |
| C.2 | (无新文件) 跑 perplexity，与 CPU 基线 diff | Δ PPL ≤ 1e-3 |
| C.3 | (无新文件) llama-bench `-ngl 0/4/8/999` × `pp512/tg128` | 数字填入 perf doc |
| C.4 | `docs/performance/unified-backend.md` 增 §"Qwen3/MiniMind-3 实跑" | 表格 + 加速比 |

**退出标准**：
- `pp512` (prefill) 在 `-ngl 999` 下比 `-ngl 0` (纯 CPU) 至少快 2×（Qwen3 64M 小模型，加速比可能更高）
- `tg128` (decode) 在 `-ngl 999` 下比 `-ngl 0` 至少快 1.5×（decode 受 memory bandwidth 限制，加速比天然低）
- perplexity 数字不变

### Phase D — 生产化硬化

**目标**：让 ggml-triton 后端从"开发能跑"到"CI 全绿"。

| 步骤 | 验证 |
|---|---|
| D.1 SM 变体补齐 | `-DGPU_ARCHS="70;75;80;86;89;90"` configure 全部成功；AOT 产物都在 |
| D.2 Provider fallback 测试 | `-DGGML_TRITON=ON -DGGML_TRITON_CPU_ONLY=ON` 和默认配置下，`test-triton-registry.cpp` 都通过 |
| D.3 失败模式文档化 | `ggml-custom-backends.md` §"已知失败模式" 有 ≥ 5 个条目（dispatch 决策表错误、AOT 缺变体、CUTLASS 版本不兼容、TileLang codegen 失败、CUDA driver 缺符号等） |
| D.4 ROCm 评估 | `docs/backend/TRITON.md` §"硬件" 加 ROCm 行，标 "评估中" 或 "不支持"（取决于实际工作量） |

**退出标准**：
- `ci/run.sh` 中 `GG_BUILD_TRITON=1` 全绿
- 所有 SM 变体 AOT 产物 check-in
- 失败模式文档化

## 4. 依赖关系

```
   A.1 (GPU host)
    │
    ├─── A.2 ── A.3 ── A.4
    │                        │
    │                        ▼
    │                     (Phase A 退出)
    │                        │
    B.1 ── B.2 ── B.3       (Phase B 可在 CPU box 上做)
    │                        │
    ▼                        │
 (Phase B 退出)              │
    │                        │
    └─────── C.1 ── C.2 ─── C.3 ── C.4
                              │
                           (Phase C 退出 = 项目目标达成)
                              │
                              ▼
                            D.1 ── D.2 ── D.3 ── D.4
                                          │
                                       (Phase D 退出 = 生产就绪)
```

关键路径：A → C。A 需要 GPU host；C 需要 A + B。B 可以在 CPU box 上做。
D 是 nice-to-have，不阻塞目标达成。

## 5. 估算工作量（粗略，按"人 × 周"）

| 阶段 | 估算 | 备注 |
|---|---|---|
| A | 0.5 周 | 主要等待 GPU host + 跑 bench；代码改动理论上 0 |
| B | 3–4 周 | B.1 RMSNorm ~3 天，B.2 RoPE ~3 天，B.3 FlashAttn ~2 周（最大块） |
| C | 1 周 | 主要在 GPU host 上调参 + 写 perf doc |
| D | 2 周 | D.1 ~3 天，D.2 ~2 天，D.3 ~1 天，D.4 看 ROCm 投入度 |
| **总计** | **~8 周** | 到 D 退出 (生产就绪) |
| **最小路径** | **~4.5 周** | 只到 C 退出 (项目目标达成) |

## 6. 风险

- **R1**：A 阶段如果 GPU host 不可得，Phase A + C 整体阻塞。**对策**：先做
  B（B 只需要 CPU box）。
- **R2**：B.3 FlashAttn 实际工作量可能比 2 周大（FP16 在线 softmax + 块化的
  Triton DSL 调试周期不可预测）。**对策**：先做 B.1 + B.2 验证流程打通，B.3
  可以拆出来独立排期。
- **R3**：Phase C 的"加速比 ≥ 2×"假设不一定成立——MiniMind-3 64M 很小，
  GPU launch overhead 可能吃掉加速。**对策**：C.3 用 `-ngl 0/4/8/all` 4 档
  看趋势，而不是只看 2 档的二分。
- **R4**：Phase D.4 ROCm 实际投入可能超过 1 周。**对策**：D.4 标记为可选，
  阻塞 D.1-D.3，但不阻塞 D 整体退出。

## 7. 与现有文档的关系

- [`README.md`](README.md) §"当前状态" 表 = 本路线图的"现状"小节
- [`ggml-custom-backends.md`](ggml-custom-backends.md) = 路线图 Phase B 的
  "操作手册"（如何加新 provider 的 4 步流程）+ §"已知失败模式" 记录 B.1
  揭示的设计权衡
- [`minimind-integration.md`](minimind-integration.md) = Phase C 的端到端
  recipe
- [`test-pyramid.md`](test-pyramid.md) = Phase A/C 的测试门禁（已
  在 B.1 commit 中标记 RMSNorm 为 covered op）
- [`../performance/unified-backend.md`](../performance/unified-backend.md)
  = Phase A 的填空区 + Phase C 的扩展区
- [`../superpowers/plans/baseline-deferral.md`](../superpowers/plans/baseline-deferral.md)
  = Phase A 的精确 recipe
- [`../superpowers/plans/2026-06-10-rmsnorm-triton-aot.md`](../superpowers/plans/2026-06-10-rmsnorm-triton-aot.md)
  = B.1 RMSNorm provider 的完整执行 plan（15 task / 2 stage），含 Oracle
  审查反馈、Stage 1 限制、Stage 2 升级路径
- [`../superpowers/plans/2026-06-09-phase-0-audit.md`](../superpowers/plans/2026-06-09-phase-0-audit.md)
  = Phase 0 环境前提审计（B.1 引用 §0.4 的 placeholder CUBIN 现实）

## 8. 何时更新本页

- ✅ Phase A/B/C/D 任何一个阶段状态从 ⏸ → ✅ → 退出时 — 更新 §1 状态表
- ✅ 风险条款（R1-R4）任何一个被触发时 — 更新 §6 状态
- ✅ 路线图分叉（例如 "B.3 拆成 B.3a + B.3b"）— 更新 §3 详细分解
- ✅ 阶段新增（例如 "Phase E：CI 集成"）— 在 §2 流程图加节点
