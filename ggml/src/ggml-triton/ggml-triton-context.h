#pragma once

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-triton-provider.h"

#ifndef GGML_TRITON_CPU_ONLY
#include <cuda.h>
#endif

#include <cstddef>
#include <mutex>
#include <string>
#include <unordered_map>

// Internal context for a single Triton stream/backend instance.
//
// The backend is intentionally minimal at this stage: kernels are AOT compiled
// from Triton into CUBINs by the build system, then loaded lazily through the
// CUDA Driver API. The maps below cache loaded modules and resolved kernel
// functions to avoid repeated cuModuleLoadData / cuModuleGetFunction work.
struct ggml_backend_triton_context {
    int                                              device_id        = 0;
    bool                                             cpu_only         = false;
    int                                              compute_capability = 0;   // e.g. 80 for SM80

    std::string                                      name;
    std::string                                      description;

#ifndef GGML_TRITON_CPU_ONLY
    CUdevice                                         cu_device        = 0;
    CUcontext                                        cu_context       = nullptr;
    CUstream                                         cu_stream        = nullptr;

    // kernel name -> loaded CUmodule (one per AOT kernel variant)
    std::unordered_map<std::string, CUmodule>        modules;
    // kernel name -> resolved CUfunction inside the corresponding module
    std::unordered_map<std::string, CUfunction>      functions;
#endif

    std::mutex                                       mutex;

    // Provider-based op registry for this backend instance
    ggml_triton_op_registry                           op_registry;
};

// Per-device buffer-type context (one per Triton device).
struct ggml_backend_triton_buffer_type_context {
    int         device_id = 0;
#ifndef GGML_TRITON_CPU_ONLY
    CUdevice    cu_device = 0;
#endif
    std::string name;
};

// Context attached to each ggml_backend_buffer allocated via Triton.
struct ggml_backend_triton_buffer_context {
#ifndef GGML_TRITON_CPU_ONLY
    CUdeviceptr dev_ptr  = 0;
#else
    void *      dev_ptr  = nullptr;
#endif
    size_t      size     = 0;
    int         device_id = 0;
#ifndef GGML_TRITON_CPU_ONLY
    CUcontext   cu_context = nullptr;
#endif
    std::string name;
};
