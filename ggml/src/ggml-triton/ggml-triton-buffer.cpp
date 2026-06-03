#include "ggml-triton-context.h"

#include "ggml-impl.h"
#include "ggml-backend-impl.h"

#ifndef GGML_TRITON_CPU_ONLY
#include <cuda.h>
#endif

#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <new>
#include <string>
#include <unordered_map>

// ----------------------------------------------------------------------------
// CUDA driver helpers (GPU mode only)
// ----------------------------------------------------------------------------

#ifndef GGML_TRITON_CPU_ONLY

#define TRITON_CU_CHECK(expr)                                              \
    do {                                                                   \
        CUresult _res = (expr);                                            \
        if (_res != CUDA_SUCCESS) {                                        \
            const char * _name = nullptr;                                  \
            cuGetErrorName(_res, &_name);                                  \
            GGML_LOG_ERROR("ggml-triton: %s failed: %s\n",                 \
                           #expr, _name ? _name : "unknown");              \
        }                                                                  \
    } while (0)

static void triton_push_ctx(CUcontext c) {
    if (c) {
        TRITON_CU_CHECK(cuCtxPushCurrent(c));
    }
}
static void triton_pop_ctx(CUcontext c) {
    if (c) {
        CUcontext popped = nullptr;
        TRITON_CU_CHECK(cuCtxPopCurrent(&popped));
    }
}

#endif // GGML_TRITON_CPU_ONLY

// ----------------------------------------------------------------------------
// buffer interface
// ----------------------------------------------------------------------------

static void ggml_backend_triton_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    auto * ctx = static_cast<ggml_backend_triton_buffer_context *>(buffer->context);
    if (!ctx) return;

#ifndef GGML_TRITON_CPU_ONLY
    triton_push_ctx(ctx->cu_context);
    if (ctx->dev_ptr) {
        TRITON_CU_CHECK(cuMemFree(ctx->dev_ptr));
        ctx->dev_ptr = 0;
    }
    triton_pop_ctx(ctx->cu_context);

    // Balance the cuDevicePrimaryCtxRetain() done in alloc_buffer.
    CUdevice cu_dev = 0;
    if (cuDeviceGet(&cu_dev, ctx->device_id) == CUDA_SUCCESS) {
        cuDevicePrimaryCtxRelease(cu_dev);
    }
#else
    if (ctx->dev_ptr) {
        free(ctx->dev_ptr);
        ctx->dev_ptr = nullptr;
    }
#endif
    delete ctx;
}

static void * ggml_backend_triton_buffer_get_base(ggml_backend_buffer_t buffer) {
    auto * ctx = static_cast<ggml_backend_triton_buffer_context *>(buffer->context);
    return reinterpret_cast<void *>(ctx->dev_ptr);
}

static enum ggml_status ggml_backend_triton_buffer_init_tensor(ggml_backend_buffer_t buffer,
                                                               struct ggml_tensor *  tensor) {
    GGML_UNUSED(buffer);
    GGML_UNUSED(tensor);
    return GGML_STATUS_SUCCESS;
}

static void ggml_backend_triton_buffer_memset_tensor(ggml_backend_buffer_t buffer,
                                                     struct ggml_tensor *  tensor,
                                                     uint8_t               value,
                                                     size_t                offset,
                                                     size_t                size) {
    auto * ctx = static_cast<ggml_backend_triton_buffer_context *>(buffer->context);
#ifndef GGML_TRITON_CPU_ONLY
    triton_push_ctx(ctx->cu_context);
    TRITON_CU_CHECK(cuMemsetD8((CUdeviceptr)((char *) tensor->data + offset), value, size));
    triton_pop_ctx(ctx->cu_context);
#else
    GGML_UNUSED(ctx);
    memset((char *) tensor->data + offset, value, size);
#endif
}

static void ggml_backend_triton_buffer_set_tensor(ggml_backend_buffer_t buffer,
                                                  struct ggml_tensor *  tensor,
                                                  const void *          data,
                                                  size_t                offset,
                                                  size_t                size) {
    auto * ctx = static_cast<ggml_backend_triton_buffer_context *>(buffer->context);
#ifndef GGML_TRITON_CPU_ONLY
    triton_push_ctx(ctx->cu_context);
    TRITON_CU_CHECK(cuMemcpyHtoD((CUdeviceptr)((char *) tensor->data + offset), data, size));
    triton_pop_ctx(ctx->cu_context);
#else
    GGML_UNUSED(ctx);
    memcpy((char *) tensor->data + offset, data, size);
#endif
}

static void ggml_backend_triton_buffer_get_tensor(ggml_backend_buffer_t       buffer,
                                                  const struct ggml_tensor *  tensor,
                                                  void *                      data,
                                                  size_t                      offset,
                                                  size_t                      size) {
    auto * ctx = static_cast<ggml_backend_triton_buffer_context *>(buffer->context);
#ifndef GGML_TRITON_CPU_ONLY
    triton_push_ctx(ctx->cu_context);
    TRITON_CU_CHECK(cuMemcpyDtoH(data, (CUdeviceptr)((const char *) tensor->data + offset), size));
    triton_pop_ctx(ctx->cu_context);
#else
    GGML_UNUSED(ctx);
    memcpy(data, (const char *) tensor->data + offset, size);
#endif
}

static void ggml_backend_triton_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    auto * ctx = static_cast<ggml_backend_triton_buffer_context *>(buffer->context);
#ifndef GGML_TRITON_CPU_ONLY
    triton_push_ctx(ctx->cu_context);
    TRITON_CU_CHECK(cuMemsetD8(ctx->dev_ptr, value, ctx->size));
    triton_pop_ctx(ctx->cu_context);
#else
    memset(ctx->dev_ptr, value, ctx->size);
#endif
}

static const ggml_backend_buffer_i ggml_backend_triton_buffer_interface = {
    /* .free_buffer     = */ ggml_backend_triton_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_triton_buffer_get_base,
    /* .init_tensor     = */ ggml_backend_triton_buffer_init_tensor,
    /* .memset_tensor   = */ ggml_backend_triton_buffer_memset_tensor,
    /* .set_tensor      = */ ggml_backend_triton_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_triton_buffer_get_tensor,
    /* .set_tensor_2d   = */ NULL,
    /* .get_tensor_2d   = */ NULL,
    /* .cpy_tensor      = */ NULL,
    /* .clear           = */ ggml_backend_triton_buffer_clear,
    /* .reset           = */ NULL,
};

// ----------------------------------------------------------------------------
// buffer-type interface
// ----------------------------------------------------------------------------

static const char * ggml_backend_triton_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    auto * ctx = static_cast<ggml_backend_triton_buffer_type_context *>(buft->context);
    return ctx->name.c_str();
}

bool ggml_backend_buft_is_triton(ggml_backend_buffer_type_t buft) {
    return buft != nullptr && buft->iface.get_name == ggml_backend_triton_buffer_type_get_name;
}

static ggml_backend_buffer_t ggml_backend_triton_buffer_type_alloc_buffer(
        ggml_backend_buffer_type_t buft, size_t size) {
    auto * buft_ctx = static_cast<ggml_backend_triton_buffer_type_context *>(buft->context);

#ifndef GGML_TRITON_CPU_ONLY
    // Make sure we have a primary context current for this device.
    CUcontext primary = nullptr;
    CUresult  rc      = cuDevicePrimaryCtxRetain(&primary, buft_ctx->cu_device);
    if (rc != CUDA_SUCCESS) {
        const char * name = nullptr;
        cuGetErrorName(rc, &name);
        GGML_LOG_ERROR("ggml-triton: cuDevicePrimaryCtxRetain failed: %s\n", name ? name : "unknown");
        return nullptr;
    }

    triton_push_ctx(primary);

    CUdeviceptr dev_ptr = 0;
    rc = cuMemAlloc(&dev_ptr, size);
    triton_pop_ctx(primary);

    if (rc != CUDA_SUCCESS) {
        const char * name = nullptr;
        cuGetErrorName(rc, &name);
        GGML_LOG_ERROR("ggml-triton: cuMemAlloc(%.2f MiB) on device %d failed: %s\n",
                       size / 1024.0 / 1024.0, buft_ctx->device_id, name ? name : "unknown");
        cuDevicePrimaryCtxRelease(buft_ctx->cu_device);
        return nullptr;
    }

    auto * ctx = new (std::nothrow) ggml_backend_triton_buffer_context();
    if (!ctx) {
        cuMemFree(dev_ptr);
        cuDevicePrimaryCtxRelease(buft_ctx->cu_device);
        return nullptr;
    }
    ctx->dev_ptr    = dev_ptr;
    ctx->size       = size;
    ctx->device_id  = buft_ctx->device_id;
    ctx->cu_context = primary;
    ctx->name       = buft_ctx->name;
#else
    // CPU-only mode: allocate with malloc
    void * ptr = nullptr;
    if (size > 0) {
        ptr = malloc(size);
    }
    if (ptr == nullptr && size > 0) {
        GGML_LOG_ERROR("ggml-triton: malloc(%.2f MiB) failed\n",
                       size / 1024.0 / 1024.0);
        return nullptr;
    }

    auto * ctx = new (std::nothrow) ggml_backend_triton_buffer_context();
    if (!ctx) {
        free(ptr);
        return nullptr;
    }
    ctx->dev_ptr    = ptr;
    ctx->size       = size;
    ctx->device_id  = buft_ctx->device_id;
    ctx->name       = buft_ctx->name;
#endif

    return ggml_backend_buffer_init(buft, ggml_backend_triton_buffer_interface, ctx, size);
}

static size_t ggml_backend_triton_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    return 128;
}

static bool ggml_backend_triton_buffer_type_is_host(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
#ifdef GGML_TRITON_CPU_ONLY
    return true;
#else
    return false;
#endif
}

static const ggml_backend_buffer_type_i ggml_backend_triton_buffer_type_interface = {
    /* .get_name         = */ ggml_backend_triton_buffer_type_get_name,
    /* .alloc_buffer     = */ ggml_backend_triton_buffer_type_alloc_buffer,
    /* .get_alignment    = */ ggml_backend_triton_buffer_type_get_alignment,
    /* .get_max_size     = */ NULL,
    /* .get_alloc_size   = */ NULL,
    /* .is_host          = */ ggml_backend_triton_buffer_type_is_host,
};

// Public accessor used by the device interface in ggml-triton.cpp.
ggml_backend_buffer_type_t ggml_backend_triton_buffer_type(ggml_backend_dev_t dev, int device_id) {
    static std::mutex                          mutex;
    static std::unordered_map<int, ggml_backend_buffer_type> cache;

    std::lock_guard<std::mutex> lock(mutex);

    auto it = cache.find(device_id);
    if (it != cache.end()) {
        return &it->second;
    }

    auto * ctx = new ggml_backend_triton_buffer_type_context();
    ctx->device_id = device_id;
#ifndef GGML_TRITON_CPU_ONLY
    CUdevice cu_dev = 0;
    if (cuDeviceGet(&cu_dev, device_id) != CUDA_SUCCESS) {
        return nullptr;
    }
    ctx->cu_device = cu_dev;
#endif
    ctx->name      = std::string("Triton") + std::to_string(device_id);

    ggml_backend_buffer_type buft = {
        /* .iface   = */ ggml_backend_triton_buffer_type_interface,
        /* .device  = */ dev,
        /* .context = */ ctx,
    };

    auto inserted = cache.emplace(device_id, buft).first;
    return &inserted->second;
}
