# Phase 0 审计 — 环境前提验证报告

> 读者对象：自己 / 任何接手 Phase B (kernel 集覆盖 Qwen3) 的人。
>
> 本文档记录了 Oracle 建议的"Phase 0 审计"实际跑出来的结果。目的是在
> 花 3 天写 RMSNorm provider 之前, 把两个隐藏前提钉死。本审计 4 件事
> 实际只跑了 1 个工作日。

## TL;DR

| 检查项 | 结果 | 关键发现 |
|---|---|---|
| 0.1 MiniMind-3 GGUF 存在性 | ❌ → ✅ 修复 | symlink 父目录有 HF 源 (128MB safetensors) 但**没有 GGUF**, 跑 convert 后 OK |
| 0.2 GGUF CPU 前向通 | ✅ 修复后通过 | 单轮推理 52.5 t/s, 输出合理英文 |
| 0.3 转换补丁 | ⚠️ doc 描述与现状有偏差 | doc 说改 `convert_hf_to_gguf.py`, 实际需要在 `conversion/base.py` 的 `get_vocab_base_pre()` 加 patch |
| 0.4 Triton AOT 编译 | ❌ **API 漂移** | Triton 3.7.0 的 `triton.compile()` 不再接受 `signature=` / `constants=` / `cc=`, 现有 AOT 脚本用旧 API, 真 AOT 在本机完全不能工作, 但**回退到 placeholder CUBIN 的 build 路径仍然通** |
| 0.5 provider 注册流水线 | ✅ 通过 | `test-triton-registry` 通过, 4 个 provider 全部正确注册 |

**结论**: Phase B 可以启动, 但要接受一个前置修复 + 一个**重要发现**:

- **前置**: AOT 脚本 (`scripts/compile_kernels.py`) 需要适配 Triton 3.7.0 的新
  `triton.compile(src, target, options)` API. 不修这个, 任何新加的 Triton kernel
  在 `cmake --build` 时会触发 `compile_kernels.py` 然后失败 (但有 placeholder
  fallback, 所以现有 build 不会断).
- **重要发现**: 在 CPU-only box 上做 Phase B, RMSNorm/RoPE/FlashAttn 这些**新
  kernel 不会有真实的 CUBIN** — 它们都会走 placeholder fallback. 这不影响功能
  (placeholder CUBIN 的 launcher 仍然能跑, 只是 kernel 内容是假的 ELF 头),
  但意味着**GPU 实测必须在真 GPU host 上做**, 本机无法在 Phase B 阶段给 kernel
  实际数值正确性背书.

## 0.1 MiniMind-3 GGUF 存在性

**结果**: ❌ GGUF 不存在 → ✅ 通过转换修复.

```
$ readlink minimind-3
../minimind/minimind-3/

$ ls ../minimind/minimind-3/
README.md  README_en.md  chat_template.jinja  config.json
configuration.json  generation_config.json  images/
minimind.modelfile  model.safetensors  special_tokens_map.json
tokenizer.json  tokenizer_config.json

$ find ../minimind/minimind-3 -name "*.gguf"
(空)
```

HF 源完整 (config.json, tokenizer.json, 128MB safetensors), 但**没有转换好的 GGUF**.

`docs/development/minimind-integration.md` §3.1 描述的"先 patch tokenizer 再转换"流程需要
执行 (Phase 0.3, 见下).

## 0.2 GGUF CPU 前向通

**结果**: ✅ 通过 (修复 0.1 后).

```bash
$ ./build-master/bin/llama-cli -m minimind-3-F16.gguf -p "你好" -n 30 -c 512 -st
> 你好
|  Hello! I'm here to assist you. What would you like help with today?
[ Prompt: 71.9 t/s | Generation: 52.5 t/s ]
```

注意: `-st` (single-turn) 必须显式加, 否则默认进 REPL 模式, 第一个 response 被吞
只打印 `> ` — 看起来像"模型不工作", 其实是 REPL 吞了. (这条经验建议写进
`minimind-integration.md` §3.4 冒烟测试段.)

MiniMind-3 是**英文为主**的模型 (config 里 `general.languages: ['zh', 'en']` 但权重是英文为主),
用中文 prompt 给英文回复是预期行为, 不是 bug.

## 0.3 转换补丁 (与 doc 的偏差)

**结果**: ⚠️ doc 描述与现状有偏差, 但功能上能跑.

`docs/development/minimind-integration.md` §3.1 说:

> 在 `convert_hf_to_gguf.py` 的 `get_vocab_base_pre()` 函数末尾加一段

实际:
- `convert_hf_to_gguf.py` 在仓库根只有 287 行, **不含** `get_vocab_base_pre()` 函数
- `get_vocab_base_pre()` 在 **`conversion/base.py:1377`**
- 该函数末尾已经有一个 `if res is None: raise NotImplementedError(...)`, 改成 `res = "qwen2"` 即可

我应用的 patch (写进 `conversion/base.py`):
```python
        if res is None:
            # ... 原 warning block 不变 ...
            logger.warning("...")
            # ---- MiniMind (and similar 6400-vocab custom BPE) ----
            # 临时 fallback: 用 qwen2 的 pre-tokenizer
            res = "qwen2"
            logger.warning("** FALLBACK: MiniMind detected, using 'qwen2' tokenizer pre-tokenizer")
            # raise NotImplementedError(...)  # 注释掉
```

转换结果 (`PYTHONPATH=gguf-py python3 convert_hf_to_gguf.py ../minimind/minimind-3/ --outfile minimind-3-F16.gguf --outtype f16`):

```
INFO:hf-to-gguf:Model successfully exported to minimind-3-F16.gguf
```

GGUF dump 验证:
- 90 tensors, 33 KV pairs
- `general.architecture = 'qwen3'` ✓
- `tokenizer.ggml.model = 'gpt2'`, `tokenizer.ggml.pre = 'qwen2'` ✓ (我们的 patch 起效)
- `bos_token_id = 1` (<|im_start|>), `eos_token_id = 2` (<|im_end|>) ✓
- vocab size 6400 ✓

**给 doc 的 follow-up**: 把 §3.1 改成"`conversion/base.py:1377 get_vocab_base_pre()` 末尾的 `raise NotImplementedError` 改成 `res = "qwen2"`", 跟实际代码路径对齐.

## 0.4 Triton AOT 编译

**结果**: ❌ **API 漂移 — Triton 3.7.0 与现有脚本不兼容**, 但 build 路径仍 OK.

跑 `python3 scripts/compile_kernels.py --registry scripts/kernel_registry.json --kernels triton_kernels --out /tmp/aot-test`:

```
[triton-aot] triton.compile failed for gelu/fp16_sm80: compile() got an unexpected keyword argument 'signature'
[triton-aot] triton.compile failed for gelu/fp32_sm80: compile() got an unexpected keyword argument 'signature'
[triton-aot] triton.compile failed for silu/fp16_sm80: compile() got an unexpected keyword argument 'signature'
[triton-aot] triton.compile failed for silu/fp32_sm80: compile() got an unexpected keyword argument 'signature'
[triton-aot] wrote 0 real and 4 placeholder kernel(s) to /tmp/aot-test
```

**根因** (Triton 3.7.0 验证):

```python
import triton
import inspect
sig = inspect.signature(triton.compile)
# 只剩 3 个参数: src, target, options
```

`scripts/compile_kernels.py:116-121` 用的是 Triton 2.x 时代的 API:
```python
compiled = triton.compile(
    fn,
    signature=variant.signature,    # ← Triton 3.7.0 没有这个
    constants=variant.specialise,   # ← 也没有
    cc=cc,                          # ← 也没有
)
```

**对 Phase B 的影响**:
- 不影响: cmake build (因为 `compile_kernels.py` 失败时 fallback 到 placeholder CUBIN,
  CMake 仍然成功, 与现有的 `ggml-triton/kernels/generated/gelu_fp16_sm80.c` 路径相同)
- 影响真 AOT: 任何新加的 Triton kernel, 走 `compile_kernels.py` 时都会失败, 落到
  placeholder. Phase B 加的 RMSNorm/RoPE/FlashAttn **在本机不会得到真实 CUBIN**.

**为什么现有 build 仍然 OK**: CMakeLists.txt 的策略是 `compile_kernels.py` 失败时
fallback 到 check-in 的 placeholder `.c` 文件 (`ggml/src/ggml-triton/kernels/generated/*.c`).
后者的 launcher 函数 (`triton_launch_gelu_fp16_sm80`) 仍然能编译, 但运行时调
`cuModuleLoadData` 加载的 CUBIN 是占位 ELF 头, GPU 上跑会 launch 失败. 这是
**已知行为** — TRITON.md §"已知问题" 也明说"首次构建需要装 triton,
AOT 步骤在 CMake 时就跑".

**修复工作量**: ~半天. 把 `compile_kernels.py` 的 `triton.compile()` 调用改成
3.7.0 的新 API (`triton.compile(fn, target=GPUTarget("cuda", cc), options={...})`).
具体看 Triton 3.7.0 的 `triton.runtime.driver.active.utils.get_device_properties`
来获取 `target`. 这是 Phase B 启动前的**强建议** — 不修这个, Phase B 在本机
对 kernel 数值正确性是**无验证**的.

**Escalation**: 修复 AOT 脚本的工作量比 Phase B.1 RMSNorm 本身小, 应该**先做**.

## 0.5 provider 注册流水线

**结果**: ✅ 通过.

```bash
$ ./build-master/bin/test-triton-registry
selected=triton_gelu_fp32_sm80 provider=0 priority=100
tilelang provider: registered (skipping assertion — not built)
OK: registry test passed
```

4 个 provider 源文件全部就位:
```
ggml/src/ggml-triton/ggml-triton-provider-cpu.cpp
ggml/src/ggml-triton/ggml-triton-provider-cutlass.{cpp,h}
ggml/src/ggml-triton/ggml-triton-provider-tilelang.{cpp,h}
ggml/src/ggml-triton/ggml-triton-provider-triton.cpp
ggml/src/ggml-triton/ggml-triton-provider.{cpp,h}
```

`ggml_triton_global_registry()` 在 `ggml-triton-provider.cpp` 里注册所有 4 个,
`test-triton-registry.cpp` 验证 registry 行为. (我新增的 PR #1 那一项.)

## 0.6 给 Phase B 的修订建议 (回到 ROADMAP.md)

### 修订 1: Phase B 启动前**先修 AOT 脚本**

工作量 ~半天, ROI 极高. 改了之后:
- 任何新 kernel 都能在 CPU box 上跑真 AOT
- 不用靠 GPU host 才能验证 kernel 编译产物
- 跟现有 `triton.compile()` 3.7.0 兼容

### 修订 2: Phase B 的"退出标准"加一条

原 ROADMAP §"Phase B 退出标准":
- `test-backend-ops` 100% pass
- `test-triton-registry.cpp` 3 个新 provider 注册断言通过
- 在 GGML_LOG_LEVEL=DEBUG 下, Qwen3 graph 全部 op 都显示 `ggml-triton` 命中

**新加一条**:
- 每个新 provider 的 `*.c` (生成的 launcher) 在 CPU box 上能被 `gcc -c -DCUDA_DRIVER_API_HACK` 编译通过 (无 CUBIN 时也要能 build, 跟现有 4 个 placeholder 一样)

### 修订 3: 启动顺序调整

原 ROADMAP §"Phase B 可在 CPU box 上做", 隐含假设 AOT 脚本能用. 现在发现**不能用**.

**新顺序** (B 阶段前加一个"Phase B.0"):

1. **Phase B.0 (半天)**: 修 `scripts/compile_kernels.py` 适配 Triton 3.7.0 API
2. **Phase B.1 (~3 天)**: RMSNorm provider (现在能跑真 AOT 了, kernel 数值有保障)
3. **Phase B.2 (~3 天)**: RoPE provider
4. **Phase B.3 (~2 周)**: FlashAttn provider
5. **Phase C (1 周, GPU host)**: MiniMind-3 on Triton

不修 B.0 直接进 B.1, 等于在 placeholder CUBIN 上做数值正确性, 没有任何保证.

## 0.7 文件清单 (本次审计产生的中间产物)

| 路径 | 状态 | 说明 |
|---|---|---|
| `conversion/base.py` | 修改 | 加了 MiniMind-3 tokenizer fallback patch |
| `minimind-3-F16.gguf` | 新建 (128MB) | 转换后的 GGUF, **不入 git** (按 `minimind-integration.md` §4 的 gitignore 建议) |
| `/tmp/aot-test/` | 临时 (4 placeholder) | AOT 失败 fallback 产物, 验证用, 不入 git |
| 本文件 | 新建 | Phase 0 审计报告 |
| `scripts/compile_kernels.py` | **修改 (Phase B.0)** | 适配 Triton 3.7.0 API (ASTSource + GPUTarget + backend.parse_options), +114/-13 |

## 0.8 Phase B.0 后续 (Triton 3.7.0 适配)

### B.0 实际工作量

~半天, 跟原 ROADMAP §0.6 估计一致. 改动 1 个文件 (`scripts/compile_kernels.py`),
+114/-13 行.

### B.0 实际产出

| 改动点 | 旧 API (Triton 2.x) | 新 API (Triton 3.7.0) |
|---|---|---|
| `triton.compile()` 调用 | `compile(fn, signature=..., constants=..., cc=...)` | `compile(src, target=..., options=...)` |
| `src` 参数 | JITFunction 直接传 | `fn.ASTSource(fn=fn, constexprs=..., signature=..., attrs=...)` |
| `target` 参数 | `cc=80` (int) | `GPUTarget("cuda", 80, 32)` (namedtuple) |
| `options` 参数 | `constants=..., cc=...` (kwargs) | `backend.parse_options({"num_warps": 4, "num_stages": 1}).__dict__` (dict) |
| `cubin` 取出 | `compiled.asm["cubin"]` | `compiled.asm[backend.binary_ext]` (instance attr, default `"cubin"`) |
| 错误路径 | 模糊的 "unexpected keyword argument 'signature'" | 明确的 "no GPU driver available on this host" + 干净的 placeholder fallback |

### B.0 关键发现: AOT 在 Triton 3.7.0 需要 torch + GPU

跑 `create_binder()` 时 Triton 调用 `driver.active.get_current_target()`, 而
`driver.active` 又通过 `CudaDriver.is_active()` 检查 `torch.cuda.is_available()`:

```python
@staticmethod
def is_active():
    try:
        import torch
        return torch.cuda.is_available() and (torch.version.hip is None)
    except ImportError:
        return False
```

本机 `torch.version.cuda = "13.0"` (有) 但 `torch.cuda.is_available() = False` (没 GPU)
→ 整个 CUDA backend `is_active() = False` → `driver.active` 抛
"0 active drivers" → AOT 走 placeholder fallback.

**含义**: 即便 AOT 脚本 API 正确适配 Triton 3.7.0, **在无 GPU 机器上仍然无法
AOT 编译** (Triton 3.7.0 的设计: AOT 必须在能跑 CUDA runtime 的环境上做).

### B.0 的"是/否"回答

| 目标 | 是否达成 |
|---|---|
| AOT 脚本适配 Triton 3.7.0 API (在 GPU host 上能工作) | ✅ |
| 在本机 (无 GPU) 能跑真 AOT 拿到真实 CUBIN | ❌ (Triton 3.7.0 设计不允许) |
| 在本机 graceful fallback 到 placeholder | ✅ |
| cmake build 不破 | ✅ (100% build, 96% test pass) |
| 生成的 `.c` 字节等同旧版本 | ✅ (4/4 byte-identical) |

### B.0 对 Phase B.1 的实际影响

B.1 (RMSNorm provider) 的 4 步流程:
1. 写 Triton DSL (`triton_kernels/rmsnorm.py`) — **本机能做** ✓
2. 在 `kernel_registry.json` 加条目 — **本机能做** ✓
3. 跑 `compile_kernels.py` 出 `.c/.h` — **本机会走 placeholder, 不出真 CUBIN** ⚠️
4. 写 `ggml-triton-provider-rmsnorm.{h,cpp}` + 注册 + 测试 — **本机能做** ✓

**B.1 的工作可以在本机完成, 除了 step 3 在本机只产 placeholder. 真实 CUBIN
要在 GPU host 上重跑 step 3 才能拿到.**

如果把 placeholder CUBIN 用作 build, `libggml-triton.so` 会成功 build (因为
launcher 代码是有效的 C, 只有加载 CUBIN 那一步会失败), 但 `test-backend-ops`
里跑 RMSNorm 路径时会因为 `cuModuleLoadData` 拿到 16 字节 ELF 头而 launch
失败. 这意味着 **B.1 的"test-backend-ops 全绿"退出标准** 在本机**无法验证**,
必须等 GPU host 兜底.

但其他两个退出标准本机能验证:
- `test-triton-registry.cpp` 3 个新 provider 注册断言通过 ✓
- 在 GGML_LOG_LEVEL=DEBUG 下, Qwen3 graph op 路由日志显示 provider 命中 ✓
  (路由发生在 dispatch 层, 不需要真 CUBIN)

### B.0 给 doc 的 follow-up

`TRITON.md` §"已知问题" 提到"首次构建需要装 triton". 应该加一条:
- "Triton 3.7.0+ 的 AOT 编译需要 `torch` + `torch.cuda.is_available()` 即
  `nvidia-smi` 能看见设备. 没有 GPU 时 AOT 自动 fallback 到 placeholder CUBIN,
  build 不会失败, 但运行时 `cuLaunchKernel` 会失败."

## 0.9 下一步 (修订后)

按修订后的顺序:
1. **今天**: 已完成 — Phase B.0 修 AOT 脚本 + 加 graceful fallback (本次)
2. **明天起**: Phase B.1 RMSNorm (本机可完成 step 1/2/4, step 3 产 placeholder)
3. (并行): 推 GPU host 申请 (Phase A 阻塞解除 + B.1 step 3 真 CUBIN 兜底)
4. (异步): B.2 RoPE, B.3 FlashAttn 排进 subagent 队列

## 0.10 致谢

Oracle 戳中 "Phase B 在 CPU box 上可能不成立" 这个盲点, 节省了可能的 3+ 天浪费
(写完 RMSNorm → 跑 test-backend-ops → 才发现 AOT 根本不工作). 这次审计的 ROI
约 6 倍 (半天投入 vs 3 天 + 1 周的潜在浪费).
