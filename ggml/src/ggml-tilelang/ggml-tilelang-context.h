#pragma once

// Internal context structures for the ggml-tilelang backend.
//
// TileLang produces standalone CUDA source files that are compiled with nvcc
// and exposed as plain `extern "C"` launcher functions. As a result, this
// backend uses the CUDA Runtime API (cudaMalloc/cudaMemcpy/cudaStream_t/...)
// rather than the Driver API used by the Triton AOT backend.

#include "ggml.h"
#include "ggml-backend.h"

#include <cuda_runtime.h>

#include <string>

#define GGML_TILELANG_LOG_INFO(...)  fprintf(stderr, __VA_ARGS__)
#define GGML_TILELANG_LOG_WARN(...)  fprintf(stderr, __VA_ARGS__)
#define GGML_TILELANG_LOG_ERROR(...) fprintf(stderr, __VA_ARGS__)

#define TILELANG_CHECK(cmd) do {                                                    \
    cudaError_t _e = (cmd);                                                         \
    if (_e != cudaSuccess) {                                                        \
        GGML_TILELANG_LOG_ERROR("[tilelang] %s:%d %s -> %s\n",                       \
            __FILE__, __LINE__, #cmd, cudaGetErrorString(_e));                      \
        GGML_ABORT("tilelang CUDA runtime error");                                   \
    }                                                                                \
} while (0)

// Per-device backend (stream) context. One is created per ggml_backend_t.
struct ggml_backend_tilelang_context {
    int            device      = 0;
    cudaStream_t   stream      = nullptr;
    cudaDeviceProp props       = {};
    std::string    name;
    std::string    description;

    ggml_backend_tilelang_context() = default;
    explicit ggml_backend_tilelang_context(int dev);
    ~ggml_backend_tilelang_context();
};

// Per-device buffer-type context. Holds device id + display name.
struct ggml_backend_tilelang_buffer_type_context {
    int         device = 0;
    std::string name;
};

// Per-buffer context. Holds the device pointer.
struct ggml_backend_tilelang_buffer_context {
    int    device  = 0;
    void * dev_ptr = nullptr;

    ggml_backend_tilelang_buffer_context(int dev, void * ptr) : device(dev), dev_ptr(ptr) {}
    ~ggml_backend_tilelang_buffer_context();
};

// Helpers
int  ggml_backend_tilelang_get_device_count_impl(void);
void ggml_backend_tilelang_set_device(int device);

// Buffer type constructed lazily; lives in ggml-tilelang-buffer.cpp.
ggml_backend_buffer_type_t ggml_backend_tilelang_buffer_type_impl(int device);

// Buffer interface used by buffer_type alloc_buffer().
extern const ggml_backend_buffer_i ggml_backend_tilelang_buffer_interface;
