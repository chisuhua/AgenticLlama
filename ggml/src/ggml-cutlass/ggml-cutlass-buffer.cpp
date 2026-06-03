// =============================================================================
// CUDA Runtime API based device buffer for the CUTLASS backend.
//
// Implements the ggml_backend_buffer_i / ggml_backend_buffer_type_i interfaces
// using cudaMalloc/cudaFree and cudaMemcpyAsync. This is intentionally simple:
// no pooling, no virtual memory tricks - just a thin wrapper that lets the
// backend hand out device pointers to ggml-alloc.
// =============================================================================

#include <cassert>
#include <cstdio>
#include <cstring>
#include <mutex>

#include <cuda_runtime.h>

#include "ggml-impl.h"
#include "ggml-backend-impl.h"

#include "ggml-cutlass-context.h"

// -----------------------------------------------------------------------------
// Context lifetime
// -----------------------------------------------------------------------------

ggml_backend_cutlass_buffer_context::~ggml_backend_cutlass_buffer_context() {
    if (dev_ptr != nullptr) {
        ggml_cutlass::set_device(device);
        cudaError_t err = cudaFree(dev_ptr);
        if (err != cudaSuccess) {
            // Logging only; destructors must not throw.
            GGML_LOG_ERROR("CUTLASS backend: cudaFree failed: %s\n", cudaGetErrorString(err));
        }
        dev_ptr = nullptr;
    }
}

// -----------------------------------------------------------------------------
// ggml_backend_buffer_i
// -----------------------------------------------------------------------------

static void ggml_backend_cutlass_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    delete (ggml_backend_cutlass_buffer_context *) buffer->context;
}

static void * ggml_backend_cutlass_buffer_get_base(ggml_backend_buffer_t buffer) {
    auto * ctx = (ggml_backend_cutlass_buffer_context *) buffer->context;
    return ctx->dev_ptr;
}

static enum ggml_status ggml_backend_cutlass_buffer_init_tensor(
        ggml_backend_buffer_t buffer, ggml_tensor * tensor) {
    GGML_UNUSED(buffer);
    GGML_UNUSED(tensor);
    // No tensor extras required for the skeleton.
    return GGML_STATUS_SUCCESS;
}

static void ggml_backend_cutlass_buffer_memset_tensor(
        ggml_backend_buffer_t buffer, ggml_tensor * tensor,
        uint8_t value, size_t offset, size_t size) {
    auto * ctx = (ggml_backend_cutlass_buffer_context *) buffer->context;
    ggml_cutlass::set_device(ctx->device);
    GGML_CUTLASS_CHECK(cudaMemsetAsync(
        (char *) tensor->data + offset, value, size, cudaStreamPerThread));
    GGML_CUTLASS_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
}

static void ggml_backend_cutlass_buffer_set_tensor(
        ggml_backend_buffer_t buffer, ggml_tensor * tensor,
        const void * data, size_t offset, size_t size) {
    auto * ctx = (ggml_backend_cutlass_buffer_context *) buffer->context;
    ggml_cutlass::set_device(ctx->device);
    GGML_CUTLASS_CHECK(cudaMemcpyAsync(
        (char *) tensor->data + offset, data, size,
        cudaMemcpyHostToDevice, cudaStreamPerThread));
    GGML_CUTLASS_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
}

static void ggml_backend_cutlass_buffer_get_tensor(
        ggml_backend_buffer_t buffer, const ggml_tensor * tensor,
        void * data, size_t offset, size_t size) {
    auto * ctx = (ggml_backend_cutlass_buffer_context *) buffer->context;
    ggml_cutlass::set_device(ctx->device);
    GGML_CUTLASS_CHECK(cudaMemcpyAsync(
        data, (const char *) tensor->data + offset, size,
        cudaMemcpyDeviceToHost, cudaStreamPerThread));
    GGML_CUTLASS_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
}

static bool ggml_backend_cutlass_buffer_cpy_tensor(
        ggml_backend_buffer_t buffer, const ggml_tensor * src, ggml_tensor * dst) {
    GGML_UNUSED(buffer);

    // Fast path: device-to-device copy when both tensors live on (any) CUDA
    // memory. We rely on src->buffer and dst->buffer's get_base both returning
    // device pointers; cudaMemcpyAsync will sort it out via cudaMemcpyDefault.
    if (ggml_backend_buffer_is_host(src->buffer)) {
        return false;
    }

    GGML_CUTLASS_CHECK(cudaMemcpyAsync(
        dst->data, src->data, ggml_nbytes(src),
        cudaMemcpyDeviceToDevice, cudaStreamPerThread));
    GGML_CUTLASS_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
    return true;
}

static void ggml_backend_cutlass_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    auto * ctx = (ggml_backend_cutlass_buffer_context *) buffer->context;
    ggml_cutlass::set_device(ctx->device);
    GGML_CUTLASS_CHECK(cudaMemsetAsync(
        ctx->dev_ptr, value, buffer->size, cudaStreamPerThread));
    GGML_CUTLASS_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
}

static const ggml_backend_buffer_i ggml_backend_cutlass_buffer_interface = {
    /* .free_buffer    = */ ggml_backend_cutlass_buffer_free_buffer,
    /* .get_base       = */ ggml_backend_cutlass_buffer_get_base,
    /* .init_tensor    = */ ggml_backend_cutlass_buffer_init_tensor,
    /* .memset_tensor  = */ ggml_backend_cutlass_buffer_memset_tensor,
    /* .set_tensor     = */ ggml_backend_cutlass_buffer_set_tensor,
    /* .get_tensor     = */ ggml_backend_cutlass_buffer_get_tensor,
    /* .set_tensor_2d  = */ NULL,
    /* .get_tensor_2d  = */ NULL,
    /* .cpy_tensor     = */ ggml_backend_cutlass_buffer_cpy_tensor,
    /* .clear          = */ ggml_backend_cutlass_buffer_clear,
    /* .reset          = */ NULL,
};

// Whether a given buffer handle was produced by us. Used by graph_compute to
// validate tensor placement.
bool ggml_backend_buffer_is_cutlass(ggml_backend_buffer_t buffer) {
    return buffer != nullptr &&
           buffer->iface.free_buffer == ggml_backend_cutlass_buffer_free_buffer;
}

// -----------------------------------------------------------------------------
// ggml_backend_buffer_type_i
// -----------------------------------------------------------------------------

static const char * ggml_backend_cutlass_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    auto * ctx = (ggml_backend_cutlass_buffer_type_context *) buft->context;
    return ctx->name.c_str();
}

static bool ggml_backend_buft_is_cutlass(ggml_backend_buffer_type_t buft) {
    return buft != nullptr &&
           buft->iface.get_name == ggml_backend_cutlass_buffer_type_get_name;
}

static ggml_backend_buffer_t ggml_backend_cutlass_buffer_type_alloc_buffer(
        ggml_backend_buffer_type_t buft, size_t size) {
    auto * buft_ctx = (ggml_backend_cutlass_buffer_type_context *) buft->context;

    ggml_cutlass::set_device(buft_ctx->device);

    // cudaMalloc on size 0 is technically allowed but returns nullptr; round
    // up to one byte to keep ggml-alloc happy.
    const size_t alloc_size = size == 0 ? 1 : size;

    void * dev_ptr = nullptr;
    cudaError_t err = cudaMalloc(&dev_ptr, alloc_size);
    if (err != cudaSuccess) {
        // Clear the sticky error so subsequent calls don't see it.
        (void) cudaGetLastError();
        GGML_LOG_ERROR("CUTLASS backend: cudaMalloc(%.2f MiB) on device %d failed: %s\n",
                       alloc_size / 1024.0 / 1024.0,
                       buft_ctx->device,
                       cudaGetErrorString(err));
        return nullptr;
    }

    auto * ctx = new ggml_backend_cutlass_buffer_context(
        buft_ctx->device, dev_ptr, alloc_size, buft_ctx->name);

    return ggml_backend_buffer_init(
        buft, ggml_backend_cutlass_buffer_interface, ctx, size);
}

static size_t ggml_backend_cutlass_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    return GGML_CUTLASS_BUFFER_ALIGNMENT;
}

static const ggml_backend_buffer_type_i ggml_backend_cutlass_buffer_type_interface = {
    /* .get_name         = */ ggml_backend_cutlass_buffer_type_get_name,
    /* .alloc_buffer     = */ ggml_backend_cutlass_buffer_type_alloc_buffer,
    /* .get_alignment    = */ ggml_backend_cutlass_buffer_type_get_alignment,
    /* .get_max_size     = */ NULL, // defaults to SIZE_MAX
    /* .get_alloc_size   = */ NULL, // defaults to ggml_nbytes
    /* .is_host          = */ NULL,
};

// Public buffer-type accessor (also forward-declared in ggml-cutlass.h).
ggml_backend_buffer_type_t ggml_backend_cutlass_buffer_type(int device) {
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);

    if (device < 0 || device >= ggml_cutlass::device_count()) {
        return nullptr;
    }

    static ggml_backend_buffer_type buft_storage[GGML_CUTLASS_MAX_DEVICES];
    static ggml_backend_cutlass_buffer_type_context buft_ctx[GGML_CUTLASS_MAX_DEVICES];
    static bool initialized[GGML_CUTLASS_MAX_DEVICES] = {};

    if (!initialized[device]) {
        char name[64];
        snprintf(name, sizeof(name), "CUTLASS%d", device);
        buft_ctx[device].device = device;
        buft_ctx[device].name   = name;

        buft_storage[device] = ggml_backend_buffer_type{
            /* .iface   = */ ggml_backend_cutlass_buffer_type_interface,
            /* .device  = */ nullptr, // populated by the device init code
            /* .context = */ &buft_ctx[device],
        };
        initialized[device] = true;
    }

    return &buft_storage[device];
}
