// ggml-tilelang-buffer.cpp - device buffer management for the TileLang backend.
//
// Allocations go through the CUDA Runtime API (cudaMalloc / cudaFree) and
// host<->device transfers use cudaMemcpyAsync on cudaStreamPerThread.

#include "ggml-tilelang-context.h"
#include "ggml-backend-impl.h"
#include "ggml-impl.h"

#include <cuda_runtime.h>

#include <cstring>
#include <mutex>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// per-buffer interface
// ---------------------------------------------------------------------------

ggml_backend_tilelang_buffer_context::~ggml_backend_tilelang_buffer_context() {
    if (dev_ptr != nullptr) {
        ggml_backend_tilelang_set_device(device);
        cudaError_t err = cudaFree(dev_ptr);
        if (err != cudaSuccess) {
            // do not abort here: destruction must be best-effort
            GGML_TILELANG_LOG_WARN("[tilelang] cudaFree failed: %s\n", cudaGetErrorString(err));
        }
        dev_ptr = nullptr;
    }
}

static void ggml_backend_tilelang_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    auto * ctx = (ggml_backend_tilelang_buffer_context *) buffer->context;
    delete ctx;
}

static void * ggml_backend_tilelang_buffer_get_base(ggml_backend_buffer_t buffer) {
    auto * ctx = (ggml_backend_tilelang_buffer_context *) buffer->context;
    return ctx->dev_ptr;
}

static enum ggml_status ggml_backend_tilelang_buffer_init_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor) {
    GGML_UNUSED(buffer);
    GGML_UNUSED(tensor);
    return GGML_STATUS_SUCCESS;
}

static void ggml_backend_tilelang_buffer_memset_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor,
                                                       uint8_t value, size_t offset, size_t size) {
    auto * ctx = (ggml_backend_tilelang_buffer_context *) buffer->context;
    ggml_backend_tilelang_set_device(ctx->device);
    TILELANG_CHECK(cudaMemsetAsync((char *) tensor->data + offset, value, size, cudaStreamPerThread));
    TILELANG_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
}

static void ggml_backend_tilelang_buffer_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor,
                                                    const void * data, size_t offset, size_t size) {
    auto * ctx = (ggml_backend_tilelang_buffer_context *) buffer->context;
    ggml_backend_tilelang_set_device(ctx->device);
    TILELANG_CHECK(cudaMemcpyAsync((char *) tensor->data + offset, data, size,
                                   cudaMemcpyHostToDevice, cudaStreamPerThread));
    TILELANG_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
}

static void ggml_backend_tilelang_buffer_get_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * tensor,
                                                    void * data, size_t offset, size_t size) {
    auto * ctx = (ggml_backend_tilelang_buffer_context *) buffer->context;
    ggml_backend_tilelang_set_device(ctx->device);
    TILELANG_CHECK(cudaMemcpyAsync(data, (const char *) tensor->data + offset, size,
                                   cudaMemcpyDeviceToHost, cudaStreamPerThread));
    TILELANG_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
}

static bool ggml_backend_tilelang_buffer_cpy_tensor(ggml_backend_buffer_t buffer,
                                                    const ggml_tensor * src, ggml_tensor * dst) {
    GGML_UNUSED(buffer);
    // Only handle src/dst that both live in tilelang buffers (same device for now).
    if (src->buffer == nullptr || dst->buffer == nullptr) {
        return false;
    }
    if (src->buffer->iface.free_buffer != ggml_backend_tilelang_buffer_free_buffer ||
        dst->buffer->iface.free_buffer != ggml_backend_tilelang_buffer_free_buffer) {
        return false;
    }
    auto * src_ctx = (ggml_backend_tilelang_buffer_context *) src->buffer->context;
    auto * dst_ctx = (ggml_backend_tilelang_buffer_context *) dst->buffer->context;
    if (src_ctx->device != dst_ctx->device) {
        return false; // peer copies not implemented in skeleton
    }
    ggml_backend_tilelang_set_device(dst_ctx->device);
    TILELANG_CHECK(cudaMemcpyAsync(dst->data, src->data, ggml_nbytes(src),
                                   cudaMemcpyDeviceToDevice, cudaStreamPerThread));
    TILELANG_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
    return true;
}

static void ggml_backend_tilelang_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    auto * ctx = (ggml_backend_tilelang_buffer_context *) buffer->context;
    ggml_backend_tilelang_set_device(ctx->device);
    TILELANG_CHECK(cudaMemsetAsync(ctx->dev_ptr, value, buffer->size, cudaStreamPerThread));
    TILELANG_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
}

const ggml_backend_buffer_i ggml_backend_tilelang_buffer_interface = {
    /* .free_buffer    = */ ggml_backend_tilelang_buffer_free_buffer,
    /* .get_base       = */ ggml_backend_tilelang_buffer_get_base,
    /* .init_tensor    = */ ggml_backend_tilelang_buffer_init_tensor,
    /* .memset_tensor  = */ ggml_backend_tilelang_buffer_memset_tensor,
    /* .set_tensor     = */ ggml_backend_tilelang_buffer_set_tensor,
    /* .get_tensor     = */ ggml_backend_tilelang_buffer_get_tensor,
    /* .set_tensor_2d  = */ NULL,
    /* .get_tensor_2d  = */ NULL,
    /* .cpy_tensor     = */ ggml_backend_tilelang_buffer_cpy_tensor,
    /* .clear          = */ ggml_backend_tilelang_buffer_clear,
    /* .reset          = */ NULL,
};

// ---------------------------------------------------------------------------
// buffer type interface
// ---------------------------------------------------------------------------

static const char * ggml_backend_tilelang_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    auto * ctx = (ggml_backend_tilelang_buffer_type_context *) buft->context;
    return ctx->name.c_str();
}

static ggml_backend_buffer_t ggml_backend_tilelang_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    auto * buft_ctx = (ggml_backend_tilelang_buffer_type_context *) buft->context;

    ggml_backend_tilelang_set_device(buft_ctx->device);

    void * dev_ptr = nullptr;
    cudaError_t err = cudaMalloc(&dev_ptr, size);
    if (err != cudaSuccess) {
        // Drain so the next call doesn't see this error.
        (void) cudaGetLastError();
        GGML_TILELANG_LOG_ERROR("[tilelang] cudaMalloc(%zu) on device %d failed: %s\n",
                                size, buft_ctx->device, cudaGetErrorString(err));
        return nullptr;
    }

    auto * ctx = new ggml_backend_tilelang_buffer_context(buft_ctx->device, dev_ptr);
    return ggml_backend_buffer_init(buft, ggml_backend_tilelang_buffer_interface, ctx, size);
}

static size_t ggml_backend_tilelang_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    return 128; // 128-byte alignment to satisfy vectorized CUDA loads/stores
}

static size_t ggml_backend_tilelang_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, const ggml_tensor * tensor) {
    GGML_UNUSED(buft);
    return ggml_nbytes(tensor);
}

static const ggml_backend_buffer_type_i ggml_backend_tilelang_buffer_type_interface = {
    /* .get_name         = */ ggml_backend_tilelang_buffer_type_get_name,
    /* .alloc_buffer     = */ ggml_backend_tilelang_buffer_type_alloc_buffer,
    /* .get_alignment    = */ ggml_backend_tilelang_buffer_type_get_alignment,
    /* .get_max_size     = */ NULL,
    /* .get_alloc_size   = */ ggml_backend_tilelang_buffer_type_get_alloc_size,
    /* .is_host          = */ NULL,
};

ggml_backend_buffer_type_t ggml_backend_tilelang_buffer_type_impl(int device) {
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);

    const int n = ggml_backend_tilelang_get_device_count_impl();
    if (device < 0 || device >= n) {
        return nullptr;
    }

    static std::vector<ggml_backend_buffer_type> s_bufts;
    static std::vector<ggml_backend_tilelang_buffer_type_context> s_ctxs;
    static bool s_initialized = false;

    if (!s_initialized) {
        s_bufts.resize(n);
        s_ctxs.resize(n);
        for (int i = 0; i < n; ++i) {
            s_ctxs[i].device = i;
            s_ctxs[i].name   = std::string(GGML_TILELANG_NAME) + std::to_string(i);
            s_bufts[i] = {
                /* .iface   = */ ggml_backend_tilelang_buffer_type_interface,
                /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_tilelang_reg(), i),
                /* .context = */ &s_ctxs[i],
            };
        }
        s_initialized = true;
    }

    return &s_bufts[device];
}
