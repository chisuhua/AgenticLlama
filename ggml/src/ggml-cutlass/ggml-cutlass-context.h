#pragma once

#include <cstddef>
#include <string>

#include <cuda_runtime.h>

#include "ggml.h"
#include "ggml-backend.h"

// Maximum number of CUTLASS devices we will index. Mirrors GGML_CUDA_MAX_DEVICES.
#define GGML_CUTLASS_MAX_DEVICES 16

// Default workspace size (8 MiB). CUTLASS GEMM kernels typically need a small
// device-side scratch buffer for split-K reductions / persistent grids.
#define GGML_CUTLASS_DEFAULT_WORKSPACE_BYTES (8u * 1024u * 1024u)

// Memory alignment used for tensor allocations (256B is friendly to TMA / cp.async
// alignment requirements on Ampere/Hopper).
#define GGML_CUTLASS_BUFFER_ALIGNMENT 256

struct ggml_cutlass_device_info {
    int device_id        = -1;
    int compute_major    = 0;
    int compute_minor    = 0;
    int sm_count         = 0;
    int warp_size        = 32;
    size_t total_vram    = 0;
    std::string name;
    std::string pci_bus_id;
};

// Per-stream backend context. Owned by `ggml_backend::context`.
struct ggml_backend_cutlass_context {
    int             device_id = 0;
    cudaStream_t    stream    = nullptr;
    bool            owns_stream = false;

    ggml_cutlass_device_info info;

    // Device-side workspace used by GEMM kernels. Lazily allocated.
    void *  workspace      = nullptr;
    size_t  workspace_size = 0;

    // Friendly name used by ggml_backend_get_name (e.g. "CUTLASS0").
    std::string name;

    ggml_backend_cutlass_context() = default;
    ~ggml_backend_cutlass_context();

    // Make non-copyable; CUDA resources are not safely duplicable.
    ggml_backend_cutlass_context(const ggml_backend_cutlass_context &) = delete;
    ggml_backend_cutlass_context & operator=(const ggml_backend_cutlass_context &) = delete;

    // Ensure the workspace has at least `size` bytes; (re)allocates if needed.
    cudaError_t ensure_workspace(size_t size);
};

// Per-buffer (allocation) context.
struct ggml_backend_cutlass_buffer_context {
    int     device  = 0;
    void *  dev_ptr = nullptr;
    size_t  size    = 0;
    std::string name;

    ggml_backend_cutlass_buffer_context(int dev, void * ptr, size_t sz, std::string n)
        : device(dev), dev_ptr(ptr), size(sz), name(std::move(n)) {}
    ~ggml_backend_cutlass_buffer_context();
};

// Per-buffer-type context (one per device id).
struct ggml_backend_cutlass_buffer_type_context {
    int         device = 0;
    std::string name;
};

// Per-device context.
struct ggml_backend_cutlass_device_context {
    int         device = 0;
    std::string name;
    std::string description;
    std::string pci_bus_id;
};

// Helpers shared across the backend translation units.
namespace ggml_cutlass {

// Initialise device info for a given device id. Returns false if the device is
// not visible / not CUDA-capable.
bool query_device_info(int device, ggml_cutlass_device_info & info);

// Total number of CUDA-capable devices. Cached after the first call. Returns 0
// when CUDA is unavailable.
int  device_count();

// Set the active CUDA device. No-op if `device` is already current.
void set_device(int device);

} // namespace ggml_cutlass

// Lightweight error-check macro. Logs and aborts on failure.
#define GGML_CUTLASS_CHECK(call)                                                            \
    do {                                                                                    \
        cudaError_t _e = (call);                                                            \
        if (_e != cudaSuccess) {                                                            \
            GGML_LOG_ERROR("CUTLASS backend: CUDA error '%s' at %s:%d\n",                   \
                           cudaGetErrorString(_e), __FILE__, __LINE__);                     \
            GGML_ABORT("CUDA failure");                                                     \
        }                                                                                   \
    } while (0)
