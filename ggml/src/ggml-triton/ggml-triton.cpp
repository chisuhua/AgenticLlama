#include "ggml-triton.h"
#include "ggml-triton-context.h"
#include "ggml-triton-dispatch.h"
#include "ggml-triton-provider.h"

#ifdef GGML_TRITON_HAS_RMSNORM
#include "ggml-triton-provider-rmsnorm.h"
#endif

#ifdef GGML_TRITON_HAS_CUTLASS
#include "ggml-triton-provider-cutlass.h"
#endif

#include "ggml-impl.h"
#include "ggml-backend-impl.h"

#ifndef GGML_TRITON_CPU_ONLY
#include <cuda.h>
#endif

#include <cstdio>
#include <cstring>
#include <mutex>
#include <new>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// forward decls (provided by ggml-triton-buffer.cpp)
// ---------------------------------------------------------------------------

ggml_backend_buffer_type_t ggml_backend_triton_buffer_type(ggml_backend_dev_t dev, int device_id);
bool                       ggml_backend_buft_is_triton(ggml_backend_buffer_type_t buft);

// ---------------------------------------------------------------------------
// CPU-only stub: ggml_triton_register_builtin_providers
//
// In CPU-only mode, ggml-triton-provider-triton.cpp is not compiled, so we
// provide an empty stub here so that provider.cpp and init() can link.
// ---------------------------------------------------------------------------

#ifdef GGML_TRITON_CPU_ONLY
void ggml_triton_register_builtin_providers(ggml_triton_op_registry & registry) {
    GGML_UNUSED(registry);
    // No GPU providers available in CPU-only mode.
    // CPU providers are registered separately via ggml_triton_register_cpu_providers.
}
#endif

// ---------------------------------------------------------------------------
// CUDA driver helpers (GPU mode only)
// ---------------------------------------------------------------------------

#ifndef GGML_TRITON_CPU_ONLY

#define TRITON_CU_CHECK_RET(expr, ret)                                     \
    do {                                                                   \
        CUresult _r = (expr);                                              \
        if (_r != CUDA_SUCCESS) {                                          \
            const char * _name = nullptr;                                  \
            cuGetErrorName(_r, &_name);                                    \
            GGML_LOG_ERROR("ggml-triton: %s failed: %s\n",                 \
                           #expr, _name ? _name : "unknown");              \
            return ret;                                                    \
        }                                                                  \
    } while (0)

static std::once_flag g_triton_cu_init_flag;
static bool           g_triton_cu_init_ok = false;

static void triton_cu_init_once() {
    std::call_once(g_triton_cu_init_flag, []() {
        CUresult r = cuInit(0);
        g_triton_cu_init_ok = (r == CUDA_SUCCESS);
        if (!g_triton_cu_init_ok) {
            const char * name = nullptr;
            cuGetErrorName(r, &name);
            GGML_LOG_INFO("ggml-triton: cuInit() failed (%s) - backend disabled\n",
                          name ? name : "unknown");
        }
    });
}

static int triton_cu_device_count() {
    triton_cu_init_once();
    if (!g_triton_cu_init_ok) {
        return 0;
    }
    int count = 0;
    if (cuDeviceGetCount(&count) != CUDA_SUCCESS) {
        return 0;
    }
    return count;
}

#endif // GGML_TRITON_CPU_ONLY

// ---------------------------------------------------------------------------
// device_context
// ---------------------------------------------------------------------------

struct ggml_backend_triton_device_context {
    int         device_id = 0;
#ifndef GGML_TRITON_CPU_ONLY
    CUdevice    cu_device = 0;
#endif
    int         compute_capability = 0;
    std::string name;
    std::string description;
};

// ---------------------------------------------------------------------------
// backend (stream) interface
// ---------------------------------------------------------------------------

static const char * ggml_backend_triton_get_name(ggml_backend_t backend) {
    auto * ctx = static_cast<ggml_backend_triton_context *>(backend->context);
    return ctx->name.c_str();
}

static void ggml_backend_triton_free(ggml_backend_t backend) {
    auto * ctx = static_cast<ggml_backend_triton_context *>(backend->context);
    if (ctx) {
#ifndef GGML_TRITON_CPU_ONLY
        if (ctx->cu_context) {
            cuCtxPushCurrent(ctx->cu_context);
            for (auto & kv : ctx->modules) {
                cuModuleUnload(kv.second);
            }
            ctx->modules.clear();
            ctx->functions.clear();
            if (ctx->cu_stream) {
                cuStreamDestroy(ctx->cu_stream);
            }
            CUcontext popped = nullptr;
            cuCtxPopCurrent(&popped);
            cuDevicePrimaryCtxRelease(ctx->cu_device);
        }
#endif
        delete ctx;
    }
    delete backend;
}

static void ggml_backend_triton_set_tensor_async(ggml_backend_t backend,
                                                 struct ggml_tensor * tensor,
                                                 const void * data,
                                                 size_t offset,
                                                 size_t size) {
#ifndef GGML_TRITON_CPU_ONLY
    auto * ctx = static_cast<ggml_backend_triton_context *>(backend->context);
    cuCtxPushCurrent(ctx->cu_context);
    cuMemcpyHtoDAsync((CUdeviceptr)((char *) tensor->data + offset), data, size, ctx->cu_stream);
    CUcontext popped = nullptr;
    cuCtxPopCurrent(&popped);
#else
    GGML_UNUSED(backend);
    memcpy((char *) tensor->data + offset, data, size);
#endif
}

static void ggml_backend_triton_get_tensor_async(ggml_backend_t backend,
                                                 const struct ggml_tensor * tensor,
                                                 void * data,
                                                 size_t offset,
                                                 size_t size) {
#ifndef GGML_TRITON_CPU_ONLY
    auto * ctx = static_cast<ggml_backend_triton_context *>(backend->context);
    cuCtxPushCurrent(ctx->cu_context);
    cuMemcpyDtoHAsync(data, (CUdeviceptr)((const char *) tensor->data + offset), size, ctx->cu_stream);
    CUcontext popped = nullptr;
    cuCtxPopCurrent(&popped);
#else
    GGML_UNUSED(backend);
    memcpy(data, (const char *) tensor->data + offset, size);
#endif
}

static void ggml_backend_triton_synchronize(ggml_backend_t backend) {
#ifndef GGML_TRITON_CPU_ONLY
    auto * ctx = static_cast<ggml_backend_triton_context *>(backend->context);
    cuCtxPushCurrent(ctx->cu_context);
    cuStreamSynchronize(ctx->cu_stream);
    CUcontext popped = nullptr;
    cuCtxPopCurrent(&popped);
#else
    GGML_UNUSED(backend);
    // CPU-only mode: all operations are synchronous, nothing to do
#endif
}

static enum ggml_status ggml_backend_triton_graph_compute(ggml_backend_t backend,
                                                          struct ggml_cgraph * cgraph) {
    auto * ctx = static_cast<ggml_backend_triton_context *>(backend->context);

#ifndef GGML_TRITON_CPU_ONLY
    cuCtxPushCurrent(ctx->cu_context);
#endif

    enum ggml_status status = GGML_STATUS_SUCCESS;
    for (int i = 0; i < cgraph->n_nodes; ++i) {
        struct ggml_tensor * node = cgraph->nodes[i];
        if (node == nullptr) {
            continue;
        }

        enum ggml_status s = ggml_triton_dispatch_op(ctx, node);
        if (s != GGML_STATUS_SUCCESS) {
            status = s;
            break;
        }
    }

#ifndef GGML_TRITON_CPU_ONLY
    cuStreamSynchronize(ctx->cu_stream);

    CUcontext popped = nullptr;
    cuCtxPopCurrent(&popped);
#endif
    return status;
}

static const ggml_backend_i ggml_backend_triton_interface = {
    /* .get_name                = */ ggml_backend_triton_get_name,
    /* .free                    = */ ggml_backend_triton_free,
    /* .set_tensor_async        = */ ggml_backend_triton_set_tensor_async,
    /* .get_tensor_async        = */ ggml_backend_triton_get_tensor_async,
    /* .set_tensor_2d_async     = */ NULL,
    /* .get_tensor_2d_async     = */ NULL,
    /* .cpy_tensor_async        = */ NULL,
    /* .synchronize             = */ ggml_backend_triton_synchronize,
    /* .graph_plan_create       = */ NULL,
    /* .graph_plan_free         = */ NULL,
    /* .graph_plan_update       = */ NULL,
    /* .graph_plan_compute      = */ NULL,
    /* .graph_compute           = */ ggml_backend_triton_graph_compute,
    /* .event_record            = */ NULL,
    /* .event_wait              = */ NULL,
    /* .graph_optimize          = */ NULL,
};

static ggml_guid_t ggml_backend_triton_guid() {
    static ggml_guid guid = { 0x74, 0x72, 0x69, 0x74, 0x6f, 0x6e, 0x42, 0x4b,
                              0x45, 0x4e, 0x44, 0xa1, 0xb2, 0xc3, 0xd4, 0xe5 };
    return &guid;
}

// ---------------------------------------------------------------------------
// device interface
// ---------------------------------------------------------------------------

static const char * ggml_backend_triton_device_get_name(ggml_backend_dev_t dev) {
    auto * ctx = static_cast<ggml_backend_triton_device_context *>(dev->context);
    return ctx->name.c_str();
}

static const char * ggml_backend_triton_device_get_description(ggml_backend_dev_t dev) {
    auto * ctx = static_cast<ggml_backend_triton_device_context *>(dev->context);
    return ctx->description.c_str();
}

static void ggml_backend_triton_device_get_memory(ggml_backend_dev_t dev,
                                                  size_t * free, size_t * total) {
    auto * ctx = static_cast<ggml_backend_triton_device_context *>(dev->context);
    *free  = 0;
    *total = 0;

#ifndef GGML_TRITON_CPU_ONLY
    CUcontext primary = nullptr;
    if (cuDevicePrimaryCtxRetain(&primary, ctx->cu_device) != CUDA_SUCCESS) {
        return;
    }
    cuCtxPushCurrent(primary);
    cuMemGetInfo(free, total);
    CUcontext popped = nullptr;
    cuCtxPopCurrent(&popped);
    cuDevicePrimaryCtxRelease(ctx->cu_device);
#else
    GGML_UNUSED(ctx);
    // CPU-only mode: report 0 for now (no GPU memory to query)
#endif
}

static enum ggml_backend_dev_type ggml_backend_triton_device_get_type(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
#ifdef GGML_TRITON_CPU_ONLY
    return GGML_BACKEND_DEVICE_TYPE_CPU;
#else
    return GGML_BACKEND_DEVICE_TYPE_GPU;
#endif
}

static void ggml_backend_triton_device_get_props(ggml_backend_dev_t dev,
                                                 struct ggml_backend_dev_props * props) {
    auto * ctx = static_cast<ggml_backend_triton_device_context *>(dev->context);
    props->name        = ggml_backend_triton_device_get_name(dev);
    props->description = ggml_backend_triton_device_get_description(dev);
    props->type        = ggml_backend_triton_device_get_type(dev);
    props->device_id   = nullptr;
    ggml_backend_triton_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->caps = {
        /* .async                 = */ true,
        /* .host_buffer           = */ false,
        /* .buffer_from_host_ptr  = */ false,
        /* .events                = */ false,
    };
    GGML_UNUSED(ctx);
}

static ggml_backend_t ggml_backend_triton_device_init_backend(ggml_backend_dev_t dev,
                                                              const char * params) {
    GGML_UNUSED(params);
    auto * dctx = static_cast<ggml_backend_triton_device_context *>(dev->context);
    return ggml_backend_triton_init(dctx->device_id);
}

static ggml_backend_buffer_type_t ggml_backend_triton_device_get_buffer_type(ggml_backend_dev_t dev) {
    auto * dctx = static_cast<ggml_backend_triton_device_context *>(dev->context);
    return ggml_backend_triton_buffer_type(dev, dctx->device_id);
}

static bool ggml_backend_triton_device_supports_op(ggml_backend_dev_t dev,
                                                   const struct ggml_tensor * op) {
    GGML_UNUSED(dev);
    return ggml_backend_triton_supports_op(op);
}

static bool ggml_backend_triton_device_supports_buft(ggml_backend_dev_t dev,
                                                     ggml_backend_buffer_type_t buft) {
    return ggml_backend_buft_is_triton(buft) && buft->device == dev;
}

static bool ggml_backend_triton_device_offload_op(ggml_backend_dev_t dev,
                                                  const struct ggml_tensor * op) {
    GGML_UNUSED(dev);
    GGML_UNUSED(op);
    return false;
}

static const ggml_backend_device_i ggml_backend_triton_device_interface = {
    /* .get_name                = */ ggml_backend_triton_device_get_name,
    /* .get_description         = */ ggml_backend_triton_device_get_description,
    /* .get_memory              = */ ggml_backend_triton_device_get_memory,
    /* .get_type                = */ ggml_backend_triton_device_get_type,
    /* .get_props               = */ ggml_backend_triton_device_get_props,
    /* .init_backend            = */ ggml_backend_triton_device_init_backend,
    /* .get_buffer_type         = */ ggml_backend_triton_device_get_buffer_type,
    /* .get_host_buffer_type    = */ NULL,
    /* .buffer_from_host_ptr    = */ NULL,
    /* .supports_op             = */ ggml_backend_triton_device_supports_op,
    /* .supports_buft           = */ ggml_backend_triton_device_supports_buft,
    /* .offload_op              = */ ggml_backend_triton_device_offload_op,
    /* .event_new               = */ NULL,
    /* .event_free              = */ NULL,
    /* .event_synchronize       = */ NULL,
};

// ---------------------------------------------------------------------------
// reg interface
// ---------------------------------------------------------------------------

struct ggml_backend_triton_reg_context {
    std::vector<ggml_backend_dev_t> devices;
};

static const char * ggml_backend_triton_reg_get_name(ggml_backend_reg_t reg) {
    GGML_UNUSED(reg);
    return "Triton";
}

static size_t ggml_backend_triton_reg_get_device_count(ggml_backend_reg_t reg) {
    auto * ctx = static_cast<ggml_backend_triton_reg_context *>(reg->context);
    return ctx->devices.size();
}

static ggml_backend_dev_t ggml_backend_triton_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    auto * ctx = static_cast<ggml_backend_triton_reg_context *>(reg->context);
    GGML_ASSERT(index < ctx->devices.size());
    return ctx->devices[index];
}

static void * ggml_backend_triton_reg_get_proc_address(ggml_backend_reg_t reg, const char * name) {
    GGML_UNUSED(reg);
    GGML_UNUSED(name);
    return nullptr;
}

static const ggml_backend_reg_i ggml_backend_triton_reg_interface = {
    /* .get_name          = */ ggml_backend_triton_reg_get_name,
    /* .get_device_count  = */ ggml_backend_triton_reg_get_device_count,
    /* .get_device        = */ ggml_backend_triton_reg_get_device,
    /* .get_proc_address  = */ ggml_backend_triton_reg_get_proc_address,
};

// ---------------------------------------------------------------------------
// public API
// ---------------------------------------------------------------------------

int ggml_backend_triton_get_device_count(void) {
#ifdef GGML_TRITON_CPU_ONLY
    return 1;
#else
    return triton_cu_device_count();
#endif
}

ggml_backend_t ggml_backend_triton_init(int device) {
#ifdef GGML_TRITON_CPU_ONLY
    // CPU-only mode: create context without CUDA
    if (device < 0 || device >= 1) {
        GGML_LOG_ERROR("ggml-triton: invalid device %d\n", device);
        return nullptr;
    }

    auto * ctx = new ggml_backend_triton_context();
    ctx->device_id          = device;
    ctx->cpu_only           = true;
    ctx->compute_capability = 0;
    ctx->name               = std::string("Triton") + std::to_string(device);
    ctx->description        = "CPU-only";

    // Register kernel providers for this backend instance
    ggml_triton_register_builtin_providers(ctx->op_registry);
    ggml_triton_register_cpu_providers(ctx->op_registry);
#ifdef GGML_TRITON_HAS_RMSNORM
    ggml_triton_register_rmsnorm_providers(ctx->op_registry);
#endif
#ifdef GGML_TRITON_HAS_CUTLASS
    ggml_triton_register_cutlass_providers(ctx->op_registry);
#endif

    auto * backend = new ggml_backend{
        /* .guid    = */ ggml_backend_triton_guid(),
        /* .iface   = */ ggml_backend_triton_interface,
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_triton_reg(), device),
        /* .context = */ ctx,
    };
    return backend;
#else
    triton_cu_init_once();
    if (!g_triton_cu_init_ok) {
        return nullptr;
    }
    if (device < 0 || device >= triton_cu_device_count()) {
        GGML_LOG_ERROR("ggml-triton: invalid device %d\n", device);
        return nullptr;
    }

    CUdevice cu_dev = 0;
    if (cuDeviceGet(&cu_dev, device) != CUDA_SUCCESS) {
        return nullptr;
    }

    CUcontext primary = nullptr;
    if (cuDevicePrimaryCtxRetain(&primary, cu_dev) != CUDA_SUCCESS) {
        return nullptr;
    }
    cuCtxPushCurrent(primary);

    CUstream stream = nullptr;
    if (cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING) != CUDA_SUCCESS) {
        CUcontext popped = nullptr;
        cuCtxPopCurrent(&popped);
        cuDevicePrimaryCtxRelease(cu_dev);
        return nullptr;
    }

    int major = 0, minor = 0;
    cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, cu_dev);
    cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, cu_dev);

    auto * ctx = new ggml_backend_triton_context();
    ctx->device_id          = device;
    ctx->cu_device          = cu_dev;
    ctx->cu_context         = primary;
    ctx->cu_stream          = stream;
    ctx->compute_capability = major * 10 + minor;
    ctx->name               = std::string("Triton") + std::to_string(device);

    // Register kernel providers for this backend instance
    ggml_triton_register_builtin_providers(ctx->op_registry);
#ifdef GGML_TRITON_HAS_RMSNORM
    ggml_triton_register_rmsnorm_providers(ctx->op_registry);
#endif
#ifdef GGML_TRITON_HAS_CUTLASS
    ggml_triton_register_cutlass_providers(ctx->op_registry);
#endif

    char dev_name[256] = {};
    cuDeviceGetName(dev_name, sizeof(dev_name), cu_dev);
    ctx->description = dev_name;

    CUcontext popped = nullptr;
    cuCtxPopCurrent(&popped);

    auto * backend = new ggml_backend{
        /* .guid    = */ ggml_backend_triton_guid(),
        /* .iface   = */ ggml_backend_triton_interface,
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_triton_reg(), device),
        /* .context = */ ctx,
    };
    return backend;
#endif
}

ggml_backend_reg_t ggml_backend_triton_reg(void) {
    static ggml_backend_reg reg{};
    static bool             initialized = false;
    static std::mutex       mutex;

    std::lock_guard<std::mutex> lock(mutex);
    if (initialized) {
        return &reg;
    }

    auto * rctx = new ggml_backend_triton_reg_context();

#ifdef GGML_TRITON_CPU_ONLY
    // CPU-only mode: create a single CPU device entry
    auto * dctx = new ggml_backend_triton_device_context();
    dctx->device_id          = 0;
    dctx->compute_capability = 0;
    dctx->name               = "Triton0";
    dctx->description        = "CPU-only";

    auto * dev = new ggml_backend_device{
        /* .iface   = */ ggml_backend_triton_device_interface,
        /* .reg     = */ &reg,
        /* .context = */ dctx,
    };
    rctx->devices.push_back(dev);
#else
    triton_cu_init_once();
    if (g_triton_cu_init_ok) {
        int device_count = 0;
        if (cuDeviceGetCount(&device_count) != CUDA_SUCCESS) {
            device_count = 0;
        }

        for (int i = 0; i < device_count; ++i) {
            CUdevice cu_dev = 0;
            if (cuDeviceGet(&cu_dev, i) != CUDA_SUCCESS) {
                continue;
            }

            int major = 0, minor = 0;
            cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, cu_dev);
            cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, cu_dev);

            char dev_name[256] = {};
            cuDeviceGetName(dev_name, sizeof(dev_name), cu_dev);

            auto * dctx = new ggml_backend_triton_device_context();
            dctx->device_id          = i;
            dctx->cu_device          = cu_dev;
            dctx->compute_capability = major * 10 + minor;
            dctx->name               = std::string("Triton") + std::to_string(i);
            dctx->description        = dev_name;

            auto * dev = new ggml_backend_device{
                /* .iface   = */ ggml_backend_triton_device_interface,
                /* .reg     = */ &reg,
                /* .context = */ dctx,
            };
            rctx->devices.push_back(dev);
        }
    }
#endif

    reg = ggml_backend_reg{
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ ggml_backend_triton_reg_interface,
        /* .context     = */ rctx,
    };

    initialized = true;
    return &reg;
}

// ---------------------------------------------------------------------------
// dynamic loading entry points
// ---------------------------------------------------------------------------

static int ggml_backend_triton_score_impl(void) {
    return ggml_backend_triton_get_device_count() > 0 ? 50 : 0;
}

GGML_BACKEND_DL_IMPL(ggml_backend_triton_reg)
GGML_BACKEND_DL_SCORE_IMPL(ggml_backend_triton_score_impl)
