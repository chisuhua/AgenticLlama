# ggml 自定义 GPU 后端架构设计

## 1. 项目概述

### 1.1 目标

为 llama.cpp 的 ggml 引擎提供多种可插拔的 GPU kernel 实现后端，使研究人员和开发者可以利用不同的 kernel 编写技术（Triton DSL、TileLang DSL、CUTLASS C++ 模板）来加速推理。

### 1.2 三个独立后端

| 后端 | 核心技术 | 定位 |
|------|---------|------|
| **ggml-triton** | OpenAI Triton Python DSL → AOT → CUBIN | 快速原型 + 跨平台（支持 ROCm） |
| **ggml-tilelang** | TileLang Python DSL → TVM lower → CUDA source | 显式流水线控制 + BitBLAS 量化生态 |
| **ggml-cutlass** | NVIDIA CUTLASS 3.x C++ 模板 | 极致 GEMM 性能 + 原生量化支持 |

### 1.3 设计原则

- **AOT 编译**：所有后端在构建期完成 kernel 编译，运行时零 Python 依赖
- **标准 ggml 后端接口**：完整实现 `ggml_backend_reg_i` / `ggml_backend_device_i` / `ggml_backend_i` 协议
- **渐进式算子覆盖**：从少量验证算子起步，逐步扩展至全量算子
- **独立编译单元**：每个后端为独立 `.so`，不互相依赖

---

## 2. 整体架构

### 2.1 分层架构图

```
┌─────────────────────────────────────────────────────────────────┐
│                        llama.cpp 推理层                          │
├─────────────────────────────────────────────────────────────────┤
│                     ggml scheduler (图调度)                      │
│         supports_op() 查询 → 路由 → fallback to ggml-cpu        │
├────────────────┬──────────────────┬─────────────────────────────┤
│  ggml-triton   │  ggml-tilelang   │       ggml-cutlass          │
│                │                  │                             │
│ ┌────────────┐ │ ┌──────────────┐ │ ┌─────────────────────────┐ │
│ │ reg        │ │ │ reg          │ │ │ reg                     │ │
│ │ device     │ │ │ device       │ │ │ device                  │ │
│ │ buffer     │ │ │ buffer       │ │ │ buffer                  │ │
│ │ graph_comp │ │ │ graph_comp   │ │ │ graph_comp              │ │
│ │ dispatch   │ │ │ dispatch     │ │ │ dispatch                │ │
│ │ kernels    │ │ │ kernels      │ │ │ kernels                 │ │
│ └────────────┘ │ └──────────────┘ │ └─────────────────────────┘ │
├────────────────┼──────────────────┼─────────────────────────────┤
│ CUDA Driver API│ CUDA Runtime API │     CUDA Runtime API        │
│ (libcuda.so)   │ (libcudart.so)   │     (libcudart.so)          │
└────────────────┴──────────────────┴─────────────────────────────┘
```

### 2.2 与 ggml-backend 体系的集成方式

三个后端均遵循 ggml-backend v2 图级（graph-based）执行模型：

1. **图级执行**：scheduler 将完整计算图交给后端的 `graph_compute()` 一次性执行
2. **能力声明**：每个后端通过 `supports_op()` 声明自己能处理的算子集合
3. **自动路由**：scheduler 根据 `supports_op()` 将算子分配到最优后端
4. **Fallback 机制**：不被任何 GPU 后端支持的算子自动 fallback 到 `ggml-cpu`

### 2.3 动态加载协议

每个后端通过宏声明动态加载入口：

```cpp
// 注册函数 - 返回 ggml_backend_reg_t
GGML_BACKEND_DL_IMPL(ggml_backend_xxx_reg)

// 评分函数 - 返回 >0 表示可用
GGML_BACKEND_DL_SCORE_IMPL(ggml_backend_xxx_score)
```

协议要点：
- `GGML_BACKEND_API_VERSION`：确保 ABI 兼容
- 命名约定：`libggml-triton.so`、`libggml-tilelang.so`、`libggml-cutlass.so`
- Score 机制：Triton/TileLang 返回 50，CUTLASS 返回 60（更高优先级）

---

## 3. ggml-triton 后端

### 3.1 架构设计

```
构建期                                    运行期
┌──────────────────┐                    ┌───────────────────────────┐
│ Python Triton    │                    │ CUDA Driver API only      │
│ elementwise.py   │                    │                           │
│        ↓         │                    │ cuModuleLoadData(CUBIN)   │
│ Triton AOT       │                    │ cuModuleGetFunction()     │
│        ↓         │                    │ cuLaunchKernel()          │
│ CUBIN + C stub   │ ── 编译链接 ──→    │                           │
└──────────────────┘                    └───────────────────────────┘
```

- **构建期**：Python Triton → AOT 编译 → 生成 CUBIN 二进制 + C launcher 函数
- **运行期**：仅依赖 CUDA Driver API（`libcuda.so`），通过 `cuModuleLoadData` 加载 CUBIN
- **无 nvcc 依赖**：不需要 CUDA Runtime 或 nvcc 编译器

### 3.2 目录结构

```
ggml/src/ggml-triton/
├── CMakeLists.txt                      # 构建配置，链接 CUDA::cuda_driver
├── ggml-triton.cpp                     # 后端注册/设备/backend 接口实现
├── ggml-triton-buffer.cpp              # GPU 显存 buffer 管理 (cuMemAlloc)
├── ggml-triton-context.h               # 上下文定义 (CUcontext, CUstream, 模块缓存)
├── ggml-triton-dispatch.cpp            # 算子分发逻辑
├── ggml-triton-dispatch.h              # dispatch 接口声明
└── kernels/
    ├── include/
    │   └── triton_kernels.h            # 聚合头文件，引入所有 AOT launcher
    └── generated/                      # AOT 编译输出（可 check-in 也可重新生成）
        ├── gelu_fp16_sm80.c            # GELU FP16 launcher + CUBIN payload
        ├── gelu_fp16_sm80.h
        ├── gelu_fp32_sm80.c
        ├── gelu_fp32_sm80.h
        ├── silu_fp16_sm80.c
        ├── silu_fp16_sm80.h
        ├── silu_fp32_sm80.c
        └── silu_fp32_sm80.h
```

### 3.3 AOT 编译流程

```
triton_kernels/elementwise.py
        │
        ▼
scripts/compile_kernels.py --registry scripts/kernel_registry.json
        │
        ▼
ggml/src/ggml-triton/kernels/generated/*.c + *.h
```

**kernel_registry.json 配置格式**：
- 定义每个 kernel 的名称、数据类型、目标 SM 架构
- 支持多 SM 版本编译策略（如 sm80、sm86、sm90）

**AOT launcher 签名**：
```c
int triton_launch_<kernel>_<dtype>_<arch>(CUstream stream,
                                          CUdeviceptr in,
                                          CUdeviceptr out,
                                          int32_t N);
```

### 3.4 核心接口实现

| 层次 | 关键实现 |
|------|---------|
| 后端注册 (`ggml_backend_reg_i`) | `ggml_backend_triton_reg()` - 单例模式，枚举 CUDA 设备 |
| 设备管理 (`ggml_backend_device_i`) | 通过 Driver API 查询显存、compute capability |
| Buffer 管理 | `cuMemAlloc` 分配，128 字节对齐，基于 `CUdeviceptr` |
| 图计算 | `graph_compute` → 遍历节点 → `ggml_triton_dispatch_op()` |
| 数据传输 | `cuMemcpyHtoDAsync` / `cuMemcpyDtoHAsync` |

### 3.5 当前支持的算子

| 算子 | 数据类型 | 实现状态 |
|------|---------|---------|
| GELU | FP16, FP32 | ✅ 已实现 |
| SILU | FP16, FP32 | ✅ 已实现 |
| NONE/VIEW/RESHAPE/PERMUTE/TRANSPOSE/CONT | - | ✅ Pass-through |

---

## 4. ggml-tilelang 后端

### 4.1 架构设计

```
构建期                                    运行期
┌──────────────────┐                    ┌───────────────────────────┐
│ TileLang Python  │                    │ CUDA Runtime API          │
│ elementwise.py   │                    │                           │
│        ↓         │                    │ 直接函数调用              │
│ TVM lower        │                    │ (静态链接到 .so)          │
│        ↓         │                    │                           │
│ CUDA .cu source  │ ── nvcc 编译 ──→   │ <<<kernel>>>              │
└──────────────────┘                    └───────────────────────────┘
```

- **构建期**：TileLang Python DSL → TVM lower → 生成标准 CUDA source (.cu)
- **运行期**：依赖 CUDA Runtime API（`libcudart.so`）
- **需要 nvcc**：生成的 .cu 文件由 CMake 的 CUDA 语言支持编译
- **零 kernel 加载开销**：kernel 作为普通 C 函数静态链接，无需 `cuModuleLoad`

### 4.2 目录结构

```
ggml/src/ggml-tilelang/
├── CMakeLists.txt                      # 构建配置，链接 CUDA::cudart
├── ggml-tilelang.cpp                   # 后端注册/设备/backend 接口实现
├── ggml-tilelang-buffer.cpp            # GPU 显存 buffer 管理 (cudaMalloc)
├── ggml-tilelang-context.h             # 上下文定义 (cudaStream_t, cudaDeviceProp)
├── ggml-tilelang-dispatch.cpp          # 算子分发逻辑
├── ggml-tilelang-dispatch.h            # dispatch 接口声明
├── kernels/
│   ├── include/
│   │   └── tilelang_kernels.h          # extern "C" launcher 声明
│   └── generated/                      # AOT 编译输出的 .cu 文件
│       ├── add_fp16.cu
│       ├── add_fp32.cu
│       ├── mul_fp16.cu
│       └── mul_fp32.cu
├── scripts/
│   ├── compile_kernels.py              # AOT 编译驱动脚本
│   └── kernel_configs.json             # kernel 配置定义
└── tilelang_kernels/
    └── elementwise.py                  # TileLang DSL kernel 源码
```

### 4.3 AOT 编译流程

```
tilelang_kernels/elementwise.py
        │
        ▼
scripts/compile_kernels.py --config scripts/kernel_configs.json
        │
        ▼
kernels/generated/*.cu
```

**kernel_configs.json 配置格式**：
```json
{
  "kernels": [
    {
      "name": "tilelang_add_fp16",
      "op": "add",
      "dtype": "float16",
      "block_size": 1024,
      "out_file": "add_fp16.cu"
    }
  ]
}
```

### 4.4 核心接口实现

| 层次 | 关键实现 |
|------|---------|
| 后端注册 | `ggml_backend_tilelang_reg()` - 单例，`cudaGetDeviceCount` 枚举 |
| 设备管理 | `cudaGetDeviceProperties` / `cudaMemGetInfo` |
| Buffer 管理 | `cudaMalloc` / `cudaFree`，标准 CUDA Runtime |
| 图计算 | `graph_compute` → 遍历节点 → `ggml_backend_tilelang_dispatch()` |
| Kernel 调用 | 普通 C 函数调用（无 cuModuleLoad） |

### 4.5 当前支持的算子

| 算子 | 数据类型 | 实现状态 |
|------|---------|---------|
| ADD | FP16, FP32 | ✅ 已实现 |
| MUL | FP16, FP32 | ✅ 已实现 |
| NONE/VIEW/RESHAPE/PERMUTE/TRANSPOSE | - | ✅ Pass-through |

### 4.6 与 Triton 后端的关键差异

| 维度 | ggml-triton | ggml-tilelang |
|------|-------------|---------------|
| CUDA API | Driver API (libcuda.so) | Runtime API (libcudart.so) |
| Kernel 加载 | cuModuleLoadData + cuLaunchKernel | 直接函数调用（静态链接） |
| 构建依赖 | Python + Triton (可选) | Python + TileLang + nvcc |
| AOT 输出 | CUBIN 二进制 + C stub | CUDA .cu 源文件 |
| 流水线控制 | 隐式（Triton 编译器优化） | 显式（TileLang schedule） |
| AMD 支持 | 是（Triton 原生支持 ROCm） | 否 |

---

## 5. ggml-cutlass 后端

### 5.1 架构设计

```
构建期                                    运行期
┌──────────────────┐                    ┌───────────────────────────┐
│ CUTLASS 3.x      │                    │ CUDA Runtime API          │
│ (header-only)    │                    │                           │
│        ↓         │                    │ 直接函数调用              │
│ C++ 模板实例化    │                    │ (无 kernel 加载开销)      │
│        ↓         │                    │                           │
│ nvcc 编译        │ ── 链接 ──→         │ cudaLaunchKernel          │
└──────────────────┘                    └───────────────────────────┘
```

- **CUTLASS 3.x**：header-only 库，通过 git submodule 引入
- **C++ 模板实例化**：利用 `CollectiveBuilder` API 生成高效 GEMM kernel
- **可选 CuteDSL**：Python codegen 生成额外的 GEMM 实例化代码
- **Workspace 懒分配**：GEMM 所需的临时显存按需分配（默认 8 MiB）

### 5.2 目录结构

```
ggml/src/ggml-cutlass/
├── CMakeLists.txt                      # 构建配置，支持 CUTLASS headers 检测
├── ggml-cutlass.cpp                    # 后端注册/设备/backend 接口实现
├── ggml-cutlass-buffer.cpp             # GPU 显存 buffer 管理 (cudaMalloc, 256B 对齐)
├── ggml-cutlass-context.h              # 上下文定义 (workspace, device_info)
├── ggml-cutlass-dispatch.cpp           # 算子分发逻辑 (GEMM + elementwise)
├── ggml-cutlass-dispatch.h             # dispatch 接口声明
├── kernels/
│   ├── include/
│   │   ├── cutlass_kernels.h           # 公共 kernel launcher 声明
│   │   └── ggml_quant_traits.h         # ggml 量化格式 → CUTLASS 类型映射
│   ├── gemm_f16.cu                     # FP16 GEMM (CUTLASS CollectiveBuilder 或 naive)
│   └── elementwise.cu                  # GELU/SILU element-wise kernels
├── scripts/
│   └── generate_gemm_instances.py      # CuteDSL codegen 脚本 (可选)
└── third_party/
    └── cutlass/                        # CUTLASS git submodule
```

### 5.3 Kernel 实现方式

**GEMM (gemm_f16.cu)**：
- 当 `GGML_CUTLASS_HAS_HEADERS` 定义时：使用 CUTLASS `CollectiveBuilder` API 生成高效 kernel
- 否则：使用 naive CUDA fallback kernel（用于验证链路完整性）
- 支持 workspace 传递（split-K reductions / persistent grids）

**量化格式映射 (ggml_quant_traits.h)**：
- `GgmlQ4_0Traits`：32 元素/block，18 字节/block，4-bit 权重
- `GgmlQ4_1Traits`：含 min 值，20 字节/block
- `GgmlQ8_0Traits`：32 元素/block，34 字节/block，8-bit 权重
- 通过 `static_assert` 确保与 ggml 磁盘格式一致

### 5.4 核心接口实现

| 层次 | 关键实现 |
|------|---------|
| 后端注册 | `ggml_backend_cutlass_reg()` - 单例，`cudaGetDeviceCount` 枚举 |
| 设备管理 | `cudaGetDeviceProperties` / PCI Bus ID 查询 |
| Buffer 管理 | `cudaMalloc`，256 字节对齐（TMA/cp.async 友好） |
| Workspace | 懒分配，`ensure_workspace()` 按需扩容 |
| 图计算 | `graph_compute` → `ggml_cutlass::compute_op()` |
| GEMM 调用 | `cutlass_gemm_f16_sm80(A, B, C, M, N, K, ...)` |

### 5.5 当前支持的算子

| 算子 | 数据类型 | 实现状态 |
|------|---------|---------|
| MUL_MAT | FP16×FP16→FP16 | ✅ 已实现 (2D contiguous) |
| GELU | FP16, FP32 | ✅ 已实现 |
| NONE/VIEW/RESHAPE/PERMUTE/TRANSPOSE | - | ✅ Pass-through |

---

## 6. Kernel Provider 抽象层（跨后端 kernel 混合）

### 6.1 设计动机

不同 kernel 技术各有优势：
- **Triton**：开发效率最高，Flash Attention 等复杂算子表达自然
- **CUTLASS**：GEMM 性能逼近硬件极限，原生量化支持
- **TileLang**：显式流水线控制，适合 memory-bound 算子优化

在同一推理过程中，不同算子可能适合不同的 kernel 来源。Provider 抽象层允许：
- 在**同一后端进程内**混合使用不同来源的 kernel
- 避免跨后端数据搬移（所有 kernel 共享同一 GPU 显存空间和 stream）
- 用户可通过配置选择最优 kernel 实现

### 6.2 核心接口定义

```cpp
// Provider 类型枚举
enum ggml_triton_provider_type {
    GGML_TRITON_PROVIDER_TRITON,    // Triton AOT kernel
    GGML_TRITON_PROVIDER_CUTLASS,   // CUTLASS kernel (混入 Triton 后端)
    GGML_TRITON_PROVIDER_TILELANG,  // TileLang kernel (混入 Triton 后端)
    GGML_TRITON_PROVIDER_AUTO,      // 自动选择最优
};

// 单个 kernel 实现描述
struct ggml_triton_kernel_impl {
    const char *               name;       // 如 "cutlass_gemm_f16"
    ggml_triton_provider_type  provider;   // 来源标识
    bool (* supports)(const ggml_tensor * op);   // 能力检查
    bool (* execute)(ggml_backend_triton_context * ctx,
                     const ggml_tensor * node);  // 执行函数
    int priority;  // 优先级，数值越高越优先
};

// 算子注册表
struct ggml_triton_op_registry {
    // op → 多个候选实现
    std::unordered_map<enum ggml_op,
                       std::vector<ggml_triton_kernel_impl>> impls;

    // 根据 tensor 属性选择最优实现
    const ggml_triton_kernel_impl * select(const ggml_tensor * op);
};
```

### 6.3 Provider 注册机制

**编译时注册**：
- 通过 CMake 选项（`GGML_TRITON_WITH_CUTLASS`、`GGML_TRITON_WITH_TILELANG`）控制
- 条件编译决定哪些 provider 源文件参与链接

**初始化时注册**：
```cpp
void ggml_triton_register_providers(ggml_triton_op_registry & registry) {
    // 始终注册原生 Triton kernels
    registry.register_impl(GGML_OP_UNARY, {
        .name     = "triton_gelu",
        .provider = GGML_TRITON_PROVIDER_TRITON,
        .supports = triton_gelu_supports,
        .execute  = triton_gelu_execute,
        .priority = 100,
    });

#ifdef GGML_TRITON_WITH_CUTLASS
    registry.register_impl(GGML_OP_MUL_MAT, {
        .name     = "cutlass_gemm_f16",
        .provider = GGML_TRITON_PROVIDER_CUTLASS,
        .supports = cutlass_gemm_supports,
        .execute  = cutlass_gemm_execute,
        .priority = 120,  // CUTLASS GEMM 优先
    });
#endif
}
```

**运行时选择**：
- `select()` 遍历候选实现，按 priority 降序检查 `supports()`
- 第一个满足条件的实现被选中执行

### 6.4 技术可行性

混合 Driver API 和 Runtime API 的关键前提：

| 技术点 | 说明 |
|--------|------|
| CUstream ↔ cudaStream_t | 底层为相同对象，可直接互转 |
| cuMemAlloc 显存可见性 | 同一地址空间，Runtime API 可直接访问 Driver API 分配的显存 |
| 单 .so 双 API | 一个共享库可同时调用 Driver API 和 Runtime API |
| libcuda + libcudart 共存 | 两者可在同一进程中共存，NVIDIA 官方支持 |

**Stream 互操作示例**：
```cpp
// Triton 后端使用 CUstream
CUstream driver_stream = ctx->cu_stream;

// 传给 CUTLASS kernel (接受 cudaStream_t)
cudaStream_t runtime_stream = (cudaStream_t) driver_stream;
cutlass_gemm_f16_sm80(..., runtime_stream);
```

### 6.5 典型配置示例

```
算子            → Provider             优先级    理由
─────────────────────────────────────────────────────────────
MUL_MAT        → CUTLASS provider     120      极致 GEMM 性能
FLASH_ATTN     → Triton provider      110      成熟实现，表达力强
GELU/SILU      → Triton provider      100      简单算子，Triton 足够
RMS_NORM       → TileLang provider    105      显式流水线优化
ADD/MUL        → TileLang provider    100      element-wise 基础算子
```

### 6.6 CMake 集成

```cmake
# Triton 后端的 Provider 混合选项
option(GGML_TRITON_WITH_CUTLASS  "Mix CUTLASS kernels into Triton backend"  OFF)
option(GGML_TRITON_WITH_TILELANG "Mix TileLang kernels into Triton backend" OFF)

if (GGML_TRITON_WITH_CUTLASS)
    # 将 CUTLASS kernel 源文件加入 ggml-triton 编译
    target_sources(ggml-triton PRIVATE
        ${CUTLASS_KERNEL_SOURCES})
    target_link_libraries(ggml-triton PRIVATE CUDA::cudart)
    target_compile_definitions(ggml-triton PRIVATE GGML_TRITON_WITH_CUTLASS)
endif()
```

### 6.7 dispatch 改造

改造前（固定路由）：
```cpp
enum ggml_status ggml_triton_dispatch_op(ctx, node) {
    switch (node->op) {
        case GGML_OP_UNARY:
            switch (get_unary_op(node)) {
                case GGML_UNARY_OP_GELU: return triton_op_gelu(ctx, node);
            }
    }
}
```

改造后（Provider 动态选择）：
```cpp
enum ggml_status ggml_triton_dispatch_op(ctx, node) {
    // 从注册表中选择最优实现
    const auto * impl = ctx->registry.select(node);
    if (impl == nullptr) {
        return GGML_STATUS_FAILED;
    }
    return impl->execute(ctx, node) ? GGML_STATUS_SUCCESS : GGML_STATUS_FAILED;
}
```

---

## 7. 构建系统

### 7.1 CMake 选项一览

| 选项 | 默认值 | 说明 |
|------|--------|------|
| `GGML_TRITON` | OFF | 启用 Triton AOT 后端 |
| `GGML_TILELANG` | OFF | 启用 TileLang 后端 |
| `GGML_CUTLASS` | OFF | 启用 CUTLASS 后端 |
| `GGML_TRITON_WITH_CUTLASS` | OFF | 在 Triton 后端中混入 CUTLASS kernel |
| `GGML_TRITON_WITH_TILELANG` | OFF | 在 Triton 后端中混入 TileLang kernel |
| `GGML_TRITON_PRECOMPILED` | OFF | 跳过 AOT 步骤，使用预编译 kernel |
| `GGML_CUTLASS_USE_DSL` | OFF | 使用 CuteDSL Python codegen |
| `CUTLASS_DIR` | `third_party/cutlass` | CUTLASS 头文件路径 |

### 7.2 依赖关系

| 后端 | 构建依赖 | 运行时依赖 |
|------|---------|-----------|
| ggml-triton | Python + Triton (可选，仅 AOT 步骤) | libcuda.so (CUDA Driver) |
| ggml-tilelang | Python + TileLang + nvcc | libcudart.so (CUDA Runtime) |
| ggml-cutlass | nvcc + CUTLASS headers (git submodule) | libcudart.so (CUDA Runtime) |

### 7.3 构建命令示例

```bash
# 仅启用 Triton 后端
cmake -B build -DGGML_TRITON=ON
cmake --build build

# Triton + CUTLASS Provider 混合
cmake -B build -DGGML_TRITON=ON -DGGML_TRITON_WITH_CUTLASS=ON
cmake --build build

# 仅启用 CUTLASS 后端（使用 CuteDSL codegen）
cmake -B build -DGGML_CUTLASS=ON -DGGML_CUTLASS_USE_DSL=ON
cmake --build build

# 全部后端独立启用
cmake -B build -DGGML_TRITON=ON -DGGML_TILELANG=ON -DGGML_CUTLASS=ON
cmake --build build

# 使用预编译 Triton kernel（无需 Python 环境）
cmake -B build -DGGML_TRITON=ON -DGGML_TRITON_PRECOMPILED=ON
cmake --build build
```

---

## 8. 三后端技术对比

| 维度 | ggml-triton | ggml-tilelang | ggml-cutlass |
|------|-------------|---------------|--------------|
| **Kernel 语言** | Python DSL (Triton) | Python DSL (TileLang) | C++ 模板 (CUTLASS 3.x) |
| **AOT 输出** | CUBIN + C stub | CUDA .cu source | CUDA .cu source |
| **运行时 API** | CUDA Driver API | CUDA Runtime API | CUDA Runtime API |
| **Python 依赖** | 仅构建期（可选） | 仅构建期 | 可选（CuteDSL） |
| **nvcc 依赖** | 否 | 是 | 是 |
| **性能天花板** | ~95% cuBLAS | ~98% cuBLAS | ~99% cuBLAS |
| **开发效率** | 最高 | 高 | 较低 |
| **AMD ROCm 支持** | 是（Triton 原生） | 否 | 否 |
| **量化支持** | 需手动实现 | BitBLAS 生态成熟 | 原生 mixed-input GEMM |
| **流水线控制** | 隐式（编译器） | 显式（schedule API） | 显式（C++ 模板） |
| **Kernel 加载开销** | 首次需 cuModuleLoad | 零（静态链接） | 零（静态链接） |
| **Buffer 对齐** | 128 字节 | Runtime 默认 | 256 字节 (TMA 友好) |
| **Score（动态加载）** | 50 | 50 | 60 |

---

## 9. 算子覆盖路线图

### Phase 1：链路验证（当前）

验证从 Python DSL → AOT → ggml 后端 → 推理的完整链路。

| 后端 | 算子 | 状态 |
|------|------|------|
| ggml-triton | GELU (FP16/FP32) | ✅ 完成 |
| ggml-triton | SILU (FP16/FP32) | ✅ 完成 |
| ggml-tilelang | ADD (FP16/FP32) | ✅ 完成 |
| ggml-tilelang | MUL (FP16/FP32) | ✅ 完成 |
| ggml-cutlass | MUL_MAT FP16 GEMM | ✅ 完成 |
| ggml-cutlass | GELU (FP16/FP32) | ✅ 完成 |

### Phase 2：核心算子

| 类别 | 算子 | 目标后端 |
|------|------|---------|
| Element-wise | ADD, MUL, SILU, GELU | 全部 |
| Normalization | RMS_NORM, NORM | Triton + TileLang |
| GEMM | MUL_MAT FP16/FP32 | CUTLASS + TileLang |
| 量化 GEMM | MUL_MAT Q4_0/Q4_1/Q8_0 | CUTLASS |
| Positional | ROPE | Triton |
| Attention | SOFT_MAX | Triton + TileLang |

### Phase 3：注意力 + 优化

- Flash Attention v2（Triton 实现为主）
- Kernel autotuning 框架（多实现自动选择最优）
- Provider 抽象层完整实现
- 跨后端 benchmark 工具

### Phase 4：生产就绪

- 全量 ggml 算子覆盖（覆盖推理关键路径）
- 多 GPU tensor parallelism
- AMD ROCm 支持（Triton 后端）
- CI 集成：自动化编译测试 + 性能回归检测
- 文档与示例完善

---

## 10. 文件清单

### ggml-triton 后端

| 文件 | 说明 |
|------|------|
| `ggml/src/ggml-triton/CMakeLists.txt` | 构建配置，CUDA Driver API 链接 |
| `ggml/src/ggml-triton/ggml-triton.cpp` | 后端注册/设备/backend 主实现 (465 行) |
| `ggml/src/ggml-triton/ggml-triton-buffer.cpp` | 显存 buffer 管理 (236 行) |
| `ggml/src/ggml-triton/ggml-triton-context.h` | 上下文结构定义 |
| `ggml/src/ggml-triton/ggml-triton-dispatch.cpp` | 算子分发 (166 行) |
| `ggml/src/ggml-triton/ggml-triton-dispatch.h` | 分发接口声明 |
| `ggml/src/ggml-triton/kernels/include/triton_kernels.h` | AOT launcher 聚合头文件 |
| `ggml/src/ggml-triton/kernels/generated/*.c/*.h` | 8 个 AOT 生成文件 |
| `triton_kernels/elementwise.py` | Triton DSL kernel 源码 |
| `scripts/compile_kernels.py` | AOT 编译驱动脚本 |
| `scripts/kernel_registry.json` | kernel 注册配置 |

### ggml-tilelang 后端

| 文件 | 说明 |
|------|------|
| `ggml/src/ggml-tilelang/CMakeLists.txt` | 构建配置，CUDA Runtime 链接 |
| `ggml/src/ggml-tilelang/ggml-tilelang.cpp` | 后端注册/设备/backend 主实现 (429 行) |
| `ggml/src/ggml-tilelang/ggml-tilelang-buffer.cpp` | 显存 buffer 管理 (195 行) |
| `ggml/src/ggml-tilelang/ggml-tilelang-context.h` | 上下文结构定义 |
| `ggml/src/ggml-tilelang/ggml-tilelang-dispatch.cpp` | 算子分发 (115 行) |
| `ggml/src/ggml-tilelang/ggml-tilelang-dispatch.h` | 分发接口声明 |
| `ggml/src/ggml-tilelang/kernels/include/tilelang_kernels.h` | extern "C" launcher 声明 |
| `ggml/src/ggml-tilelang/kernels/generated/*.cu` | 4 个 AOT 生成的 CUDA 文件 |
| `ggml/src/ggml-tilelang/scripts/compile_kernels.py` | AOT 编译驱动 (129 行) |
| `ggml/src/ggml-tilelang/scripts/kernel_configs.json` | kernel 配置 |
| `ggml/src/ggml-tilelang/tilelang_kernels/elementwise.py` | TileLang DSL 源码 |

### ggml-cutlass 后端

| 文件 | 说明 |
|------|------|
| `ggml/src/ggml-cutlass/CMakeLists.txt` | 构建配置，CUTLASS headers 检测 |
| `ggml/src/ggml-cutlass/ggml-cutlass.cpp` | 后端注册/设备/backend 主实现 (498 行) |
| `ggml/src/ggml-cutlass/ggml-cutlass-buffer.cpp` | 显存 buffer 管理 (223 行) |
| `ggml/src/ggml-cutlass/ggml-cutlass-context.h` | 上下文与设备信息定义 (110 行) |
| `ggml/src/ggml-cutlass/ggml-cutlass-dispatch.cpp` | 算子分发含 GEMM 逻辑 (214 行) |
| `ggml/src/ggml-cutlass/ggml-cutlass-dispatch.h` | 分发接口声明 |
| `ggml/src/ggml-cutlass/kernels/include/cutlass_kernels.h` | 公共 kernel API 声明 |
| `ggml/src/ggml-cutlass/kernels/include/ggml_quant_traits.h` | 量化格式类型映射 (131 行) |
| `ggml/src/ggml-cutlass/kernels/gemm_f16.cu` | FP16 GEMM kernel (141 行) |
| `ggml/src/ggml-cutlass/kernels/elementwise.cu` | GELU/SILU element-wise (103 行) |
| `ggml/src/ggml-cutlass/scripts/generate_gemm_instances.py` | CuteDSL codegen (120 行) |

### 公共头文件

| 文件 | 说明 |
|------|------|
| `ggml/include/ggml-triton.h` | Triton 后端公共 API |
| `ggml/include/ggml-tilelang.h` | TileLang 后端公共 API |
| `ggml/include/ggml-cutlass.h` | CUTLASS 后端公共 API |
