# llama.cpp + Triton AOT 后端

Triton 后端让 llama.cpp 可以用 OpenAI 的 Triton DSL 写的 kernel，**同时在
推理时不依赖 Python 运行时**：kernel 在构建期 AOT 编译成 CUBIN 二进制，运
行时通过 CUDA Driver API `mmap` 加载。

本文是用户视角的页面（构建、模型准备、运行）。架构、AOT 流水线、Provider
抽象、kernel registry 这些深入内容见
[`../development/ggml-custom-backends.md`](../development/ggml-custom-backends.md)。

- [背景](#背景)
- [操作系统](#操作系统)
- [硬件](#硬件)
- [支持的数据类型](#支持的数据类型)
- [模型准备](#模型准备)
- [CMake 选项](#cmake-选项)
- [构建](#构建)
- [验证](#验证)
- [测试](#测试)
- [已知问题](#已知问题)

## 背景

Triton 是一个嵌在 Python 里的 GPU kernel DSL，写 attention / MLP kernel
原型用得很多。llama.cpp 的 Triton 后端在**推理时不依赖 Python 也不依赖
`triton` 包**，办法是：

1. 构建期用 `scripts/compile_kernels.py` 把每个 kernel AOT 编译成 CUBIN。
2. 把 CUBIN 以 `static const unsigned char[]` 的形式嵌入到
   `ggml/src/ggml-triton/kernels/generated/` 下生成出的 C 文件里。
3. 运行时只用 CUDA Driver API（`cuModuleLoadData` + `cuLaunchKernel`）启动
   kernel——不需要 CUDA Runtime，也不需要 nvcc 参与 llama.cpp 二进制本身的
   构建。

这样既能用高层 DSL 写 kernel，又把运行时依赖压缩到只剩 `libcuda.so`。

Triton 后端是本 fork 新增的三个 GPU 后端之一，另外两个是：

- `ggml-tilelang` —— TileLang DSL → TVM lower → CUDA source
- `ggml-cutlass` —— CUTLASS 3.x C++ 模板，没有 AOT DSL 这层

三个都遵循同一套 `ggml_backend_reg_i` / `ggml_backend_device_i` /
`ggml_backend_i` 协议，scheduler 根据 `supports_op()` 把 op 分派过去。不
被任何 GPU 后端支持的 op 自动 fallback 到 `ggml-cpu`。

完整的架构（Provider 抽象、dispatch 逻辑、动态加载协议）见
[`../development/ggml-custom-backends.md`](../development/ggml-custom-backends.md)。

## 操作系统

| OS | 状态 | 备注 |
|---|---|---|
| Linux x86_64 | 支持 | 用 CUDA 12.x 驱动测过，理论上 12.x+ 都行 |
| Linux aarch64 | 尚未验证 | `libcuda.so` 存在，欢迎补 patch |
| Windows | 尚未验证 | 驱动、Driver API 都一样；构建需要 MSVC |

## 硬件

| 设备 | 状态 | 备注 |
|---|---|---|
| NVIDIA SM 80+（Ampere / Hopper / Ada） | 支持 | `scripts/kernel_registry.json` 默认 `arch=sm80` |
| NVIDIA SM 70/75（Volta / Turing） | 未测试 | 在 registry 里加一个 `sm70` / `sm75` 变体即可 |
| AMD GPU（ROCm） | 暂不支持 | 驱动路径一样；launcher 里需要把 `<hip/hip_runtime.h>` 替换掉——待跟进 |

`scripts/kernel_registry.json` 里的 `arch` 字段决定 CUBIN 的目标架构。要
支持新的 SM，加一个新的 `variants[]` 条目就行——kernel 源码本身不用改。

## 支持的数据类型

当前覆盖范围是有意做得窄的。每个新 op 都是先经过 provider registry 才进
公开的 dispatch table。

| Op 族 | dtype | Provider |
|---|---|---|
| `gelu` | fp16, fp32 | `elementwise`（Triton） |
| `silu` | fp16, fp32 | `elementwise`（Triton） |
| `gemm`（f16, f32, q4_0, q8_0） | — | CUTLASS provider（受 `GGML_TRITON_WITH_CUTLASS=ON` 控制） |

完整 op 列表和 dispatch 表见
[`../development/ggml-custom-backends.md`](../development/ggml-custom-backends.md) §3.5。

其他 op 一律 fallback 到 `ggml-cpu`。

## 模型准备

Triton 后端吃标准 GGUF 文件——模型格式上没有任何特殊要求。从 HF 模型
转到能在 Triton 后端上跑的 GGUF，按
[`../development/minimind-integration.md`](../development/minimind-integration.md)
走就行。那一页里覆盖了 HF→GGUF 转换、自定义 vocab 的 tokenizer 兜底
patch（MiniMind-3 就是典型例子）以及冒烟测试命令。

GGUF 拿到手之后：

```bash
./build/bin/llama-cli -m model.gguf -ngl 999   # 能 offload 的都 offload
```

`-ngl` 控制有多少层 transformer 放到 GPU 上。Triton 只会加速它有注册
kernel 的 op，剩下的自动走 `ggml-cpu`。这是**故意**的——在 kernel 集还没
填满的阶段，部分 offload 是预期的工作模式。

## CMake 选项

| 选项 | 默认 | 效果 |
|---|---|---|
| `GGML_TRITON` | `OFF` | 总开关。开 `ON` 才会编这个后端。 |
| `GGML_TRITON_PRECOMPILED` | `OFF` | 跳过 AOT 步骤，直接用 `kernels/generated/` 下 check-in 的 CUBIN。CI 机器和没装 `triton` 的本地都用这个。 |
| `GGML_TRITON_CPU_ONLY` | `OFF` | 只用 CPU reference provider 来构建后端——不链 CUDA，不链 `libcuda.so`。在没 GPU 的机器上测 dispatch 逻辑时很方便。 |
| `GGML_TRITON_WITH_CUTLASS` | `OFF` | 把 CUTLASS GEMM kernel 作为 Triton 后端的一个 provider 引进来。会让构建依赖 `nvcc`。 |

CUTLASS 和 TileLang 后端是独立的 CMake target，可以任意组合：

```bash
cmake -B build \
  -DGGML_TRITON=ON \
  -DGGML_TILELANG=ON \
  -DGGML_CUTLASS=ON
```

## 构建

本地标准构建（启用 Triton 后端）：

```bash
cmake -B build \
  -DLLAMA_FATAL_WARNINGS=ON \
  -DGGML_RPC=ON \
  -DGGML_TRITON=ON
cmake --build build --config Release -j$(nproc)
```

如果 AOT 步骤（`scripts/compile_kernels.py`）找不到能用的 Triton 装
配，CMake 会清晰地报错。要回退到 check-in 的 CUBIN（CI 机器、无 GPU
的本地 builder）：

```bash
cmake -B build -DGGML_TRITON=ON -DGGML_TRITON_PRECOMPILED=ON ...
```

CPU-only 冒烟构建（不要求 `libcuda`，但 kernel 只有 reference 实现）：

```bash
cmake -B build -DGGML_TRITON=ON -DGGML_TRITON_CPU_ONLY=ON ...
```

每个后端的完整 CI 矩阵在 `ci/run.sh`；`GG_BUILD_CUTLASS=1` 和
`GG_BUILD_TILELANG=1` 是驱动重一点的自托管 CI 用的环境变量。

## 验证

`cmake --build` 通过之后，最简单的端到端检查就是加载一次再补一小段：

```bash
./build/bin/llama-cli -m model.gguf -p "hello" -n 16
```

要确认 Triton 后端真的被选上了（而不是被 fallback 到了 CPU），把环境变
量 `LLAMA_LOG_COLORS=0` 设上，初始化时看日志里有没有提到 `ggml-triton`
的那一行。Scheduler 每次为 op 做路由决定时都会在 `GGML_LOG_LEVEL=DEBUG`
下打日志。

## 测试

Triton 后端参与标准的测试金字塔。直接相关的子集：

- `test-backend-ops` ——把每个 ggml op 都跟参考实现对一遍。这是抓
  "CPU 算 X，Triton 算 X+ε" 这种**后端间发散 bug 的唯一办法**。动
  ggml op 时必跑；对 Triton 后端来说，动 kernel 源、provider、dispatch
  表时也必跑。
- `test-llama-archs` ——为每个已注册架构建图、跑一遍、重新编码。端到
  端触发 `ggml-triton` 的 `graph_compute`。
- 完整的 `ctest -L main` 见 [`../development/test-pyramid.md`](../development/test-pyramid.md)
  §"Level 1"。

Triton 专属的补充：

- 改了 `scripts/kernel_registry.json` 或 `triton_kernels/` 里的东西，
  都要重跑 AOT 然后重新构建：

  ```bash
  python scripts/compile_kernels.py --out-dir ggml/src/ggml-triton/kernels/generated
  cmake --build build --config Release -j$(nproc)
  ```

  然后跑 `test-backend-ops` 确认新变体能 round-trip。

- 新加了 **provider**（`ggml/src/ggml-triton/` 下的 `.cpp`），在
  `ggml-triton.cpp` 的 `ggml_triton_register_builtin_providers()` 里注册
  一下；如果有可测的表面就再加一条 CTest 条目，并更新
  `docs/development/ggml-custom-backends.md` §6（Provider 抽象）。

推理引擎定制开发时使用的通用测试规划见
[`../development/test-pyramid.md`](../development/test-pyramid.md)。

## 已知问题

- **首次构建需要装 `triton`**，除非开了 `-DGGML_TRITON_PRECOMPILED=ON`。
  AOT 步骤在 CMake 时就跑 `compile_kernels.py`；没 `triton` 构建会快速
  失败。`kernels/generated/` 下 check-in 的 CUBIN 就是为了这种情况准
  备的。
- **部分 offload 是设计如此**。ggml op 的覆盖当前还很少。Scheduler 对
  大部分 op 会继续走 `ggml-cpu`；Triton 后端只在它有注册 kernel 的 op
  上"赢"。
- **缺 SM 变体**。如果 registry 里没有匹配你 GPU 的 `arch`，会直接
  launch 失败而不是优雅 fallback。加个变体重跑 AOT 即可。
- **混编 CUTLASS 需要 `nvcc`**。CUTLASS provider
  （`ggml-triton-provider-cutlass.cpp` + `ggml/src/ggml-cutlass/kernels/`
  下的 `.cu`）需要 `nvcc` 和较新的 CUDA toolkit 才能构建。在没有的机
  器上不要开 `-DGGML_TRITON_WITH_CUTLASS=ON`。
- **ROCm / 非 NVIDIA GPU 还没接上**。Launcher 用的是 CUDA Driver API；
  切到 HIP/ROCm 是以后的事。

## 相关阅读

- [`../development/ggml-custom-backends.md`](../development/ggml-custom-backends.md) ——
  本文所引用的架构总览（Provider 抽象、AOT 流水线、dispatch 逻辑、
  kernel registry、构建矩阵）。
- [`../development/minimind-integration.md`](../development/minimind-integration.md) ——
  把模型喂进引擎的端到端流程（HF → GGUF）。
- [`../development/test-pyramid.md`](../development/test-pyramid.md) ——
  引擎定制时使用的 5 级测试金字塔。
- `ggml/src/ggml-triton/` ——后端源码。
- `scripts/compile_kernels.py` + `scripts/kernel_registry.json` ——AOT
  驱动 + kernel 清单。
- `triton_kernels/` ——registry 里列出的那些 kernel 的 Triton DSL 源。
- `docs/build.md` ——完整的构建选项参考。
