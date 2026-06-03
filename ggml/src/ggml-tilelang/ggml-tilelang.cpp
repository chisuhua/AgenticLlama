// ggml-tilelang.cpp - main entry point for the TileLang AOT backend.
//
// Implements the standard ggml backend protocol (reg / device / backend) and
// drives the dispatcher for each compute node in the graph.
//
// CUDA usage in this file is restricted to the CUDA Runtime API.

#include "ggml-tilelang.h"
#include "ggml-tilelang-context.h"
#include "ggml-tilelang-dispatch.h"
#include "ggml-backend-impl.h"
#include "ggml-impl.h"

#include <cuda_runtime.h>

#include <atomic>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// device helpers
// ---------------------------------------------------------------------------

int ggml_backend_tilelang_get_device_count_impl(void) {
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess) {
        // No CUDA, no devices.
        (void) cudaGetLastError();
        return 0;
    }
    if (count > GGML_TILELANG_MAX_DEVICES) {
        count = GGML_TILELANG_MAX_DEVICES;
    }
    return count;
}

void ggml_backend_tilelang_set_device(int device) {
    static thread_local int s_current = -1;
    if (s_current != device) {
        TILELANG_CHECK(cudaSetDevice(device));
        s_current = device;
    }
}

// ---------------------------------------------------------------------------
// per-backend (stream) context
// ---------------------------------------------------------------------------

ggml_backend_tilelang_context::ggml_backend_tilelang_context(int dev) : device(dev) {
    ggml_backend_tilelang_set_device(device);
    TILELANG_CHECK(cudaGetDeviceProperties(&props, device));

    cudaError_t err = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    if (err != cudaSuccess) {
        (void) cudaGetLastError();
        GGML_TILELANG_LOG_ERROR("[tilelang] cudaStreamCreate failed on device %d: %s\n",
                                device, cudaGetErrorString(err));
        stream = nullptr;
    }

    name        = std::string(GGML_TILELANG_NAME) + std::to_string(device);
    description = props.name;
}

ggml_backend_tilelang_context::~ggml_backend_tilelang_context() {
    if (stream != nullptr) {
        ggml_backend_tilelang_set_device(device);
        cudaError_t err = cudaStreamSynchronize(stream);
        if (err != cudaSuccess) {
            (void) cudaGetLastError();
        }
        cudaStreamDestroy(stream);
        stream = nullptr;
    }
}

// ---------------------------------------------------------------------------
// backend (stream) interface
// ---------------------------------------------------------------------------

static const char * ggml_backend_tilelang_get_name(ggml_backend_t backend) {
    auto * ctx = (ggml_backend_tilelang_context *) backend->context;
    return ctx->name.c_str();
}

static void ggml_backend_tilelang_free(ggml_backend_t backend) {
    auto * ctx = (ggml_backend_tilelang_context *) backend->context;
    delete ctx;
    delete backend;
}

static void ggml_backend_tilelang_synchronize(ggml_backend_t backend) {
    auto * ctx = (ggml_backend_tilelang_context *) backend->context;
    if (ctx->stream != nullptr) {
        ggml_backend_tilelang_set_device(ctx->device);
        TILELANG_CHECK(cudaStreamSynchronize(ctx->stream));
    }
}

static enum ggml_status ggml_backend_tilelang_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    auto * ctx = (ggml_backend_tilelang_context *) backend->context;
    ggml_backend_tilelang_set_device(ctx->device);

    for (int i = 0; i < cgraph->n_nodes; ++i) {
        struct ggml_tensor * node = cgraph->nodes[i];

        if ((node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            continue;
        }

        switch (node->op) {
            case GGML_OP_NONE:
            case GGML_OP_RESHAPE:
            case GGML_OP_VIEW:
            case GGML_OP_PERMUTE:
            case GGML_OP_TRANSPOSE:
                continue;
            default:
                break;
        }

        enum ggml_status st = ggml_backend_tilelang_dispatch(node, ctx->stream);
        if (st != GGML_STATUS_SUCCESS) {
            return st;
        }
    }

    // Make graph_compute synchronous w.r.t. callers (matches BLAS behavior).
    if (ctx->stream != nullptr) {
        TILELANG_CHECK(cudaStreamSynchronize(ctx->stream));
    }
    return GGML_STATUS_SUCCESS;
}

static struct ggml_backend_i ggml_backend_tilelang_iface = {
    /* .get_name             = */ ggml_backend_tilelang_get_name,
    /* .free                 = */ ggml_backend_tilelang_free,
    /* .set_tensor_async     = */ NULL,
    /* .get_tensor_async     = */ NULL,
    /* .set_tensor_2d_async  = */ NULL,
    /* .get_tensor_2d_async  = */ NULL,
    /* .cpy_tensor_async     = */ NULL,
    /* .synchronize          = */ ggml_backend_tilelang_synchronize,
    /* .graph_plan_create    = */ NULL,
    /* .graph_plan_free      = */ NULL,
    /* .graph_plan_update    = */ NULL,
    /* .graph_plan_compute   = */ NULL,
    /* .graph_compute        = */ ggml_backend_tilelang_graph_compute,
    /* .event_record         = */ NULL,
    /* .event_wait           = */ NULL,
    /* .graph_optimize       = */ NULL,
};

static ggml_guid_t ggml_backend_tilelang_guid(void) {
    static ggml_guid guid = { 0x71, 0x4c, 0x4e, 0x47, 0x54, 0x49, 0x4c, 0x45,
                              0x2d, 0x42, 0x41, 0x43, 0x4b, 0x45, 0x4e, 0x44 };
    return &guid;
}

bool ggml_backend_is_tilelang(ggml_backend_t backend) {
    return backend != nullptr && ggml_guid_matches(backend->guid, ggml_backend_tilelang_guid());
}

// ---------------------------------------------------------------------------
// device interface
// ---------------------------------------------------------------------------

struct ggml_backend_tilelang_device_context {
    int         device      = 0;
    std::string name;
    std::string description;
};

static const char * ggml_backend_tilelang_device_get_name(ggml_backend_dev_t dev) {
    auto * ctx = (ggml_backend_tilelang_device_context *) dev->context;
    return ctx->name.c_str();
}

static const char * ggml_backend_tilelang_device_get_description(ggml_backend_dev_t dev) {
    auto * ctx = (ggml_backend_tilelang_device_context *) dev->context;
    return ctx->description.c_str();
}

static void ggml_backend_tilelang_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    auto * ctx = (ggml_backend_tilelang_device_context *) dev->context;
    ggml_backend_tilelang_set_device(ctx->device);
    cudaError_t err = cudaMemGetInfo(free, total);
    if (err != cudaSuccess) {
        (void) cudaGetLastError();
        *free = 0;
        *total = 0;
    }
}

static enum ggml_backend_dev_type ggml_backend_tilelang_device_get_type(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    return GGML_BACKEND_DEVICE_TYPE_GPU;
}

static void ggml_backend_tilelang_device_get_props(ggml_backend_dev_t dev, struct ggml_backend_dev_props * props) {
    props->name        = ggml_backend_tilelang_device_get_name(dev);
    props->description = ggml_backend_tilelang_device_get_description(dev);
    props->type        = ggml_backend_tilelang_device_get_type(dev);
    ggml_backend_tilelang_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->device_id   = nullptr;
    props->caps = {
        /* .async                = */ true,
        /* .host_buffer          = */ false,
        /* .buffer_from_host_ptr = */ false,
        /* .events               = */ false,
    };
}

static ggml_backend_t ggml_backend_tilelang_device_init(ggml_backend_dev_t dev, const char * params) {
    GGML_UNUSED(params);
    auto * dctx = (ggml_backend_tilelang_device_context *) dev->context;
    return ggml_backend_tilelang_init(dctx->device);
}

static ggml_backend_buffer_type_t ggml_backend_tilelang_device_get_buffer_type(ggml_backend_dev_t dev) {
    auto * dctx = (ggml_backend_tilelang_device_context *) dev->context;
    return ggml_backend_tilelang_buffer_type_impl(dctx->device);
}

static bool ggml_backend_tilelang_device_supports_op(ggml_backend_dev_t dev, const struct ggml_tensor * op) {
    GGML_UNUSED(dev);
    // identity / metadata-only ops always pass through
    switch (op->op) {
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
            return true;
        default:
            break;
    }
    return ggml_backend_tilelang_supports_op(op);
}

static bool ggml_backend_tilelang_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    auto * dctx = (ggml_backend_tilelang_device_context *) dev->context;
    if (buft == nullptr || buft->iface.get_name == nullptr) {
        return false;
    }
    // Accept only our own buffer type for the same device.
    ggml_backend_buffer_type_t own = ggml_backend_tilelang_buffer_type_impl(dctx->device);
    return buft == own;
}

static const struct ggml_backend_device_i ggml_backend_tilelang_device_iface = {
    /* .get_name             = */ ggml_backend_tilelang_device_get_name,
    /* .get_description      = */ ggml_backend_tilelang_device_get_description,
    /* .get_memory           = */ ggml_backend_tilelang_device_get_memory,
    /* .get_type             = */ ggml_backend_tilelang_device_get_type,
    /* .get_props            = */ ggml_backend_tilelang_device_get_props,
    /* .init_backend         = */ ggml_backend_tilelang_device_init,
    /* .get_buffer_type      = */ ggml_backend_tilelang_device_get_buffer_type,
    /* .get_host_buffer_type = */ NULL,
    /* .buffer_from_host_ptr = */ NULL,
    /* .supports_op          = */ ggml_backend_tilelang_device_supports_op,
    /* .supports_buft        = */ ggml_backend_tilelang_device_supports_buft,
    /* .offload_op           = */ NULL,
    /* .event_new            = */ NULL,
    /* .event_free           = */ NULL,
    /* .event_synchronize    = */ NULL,
};

// ---------------------------------------------------------------------------
// reg interface
// ---------------------------------------------------------------------------

struct ggml_backend_tilelang_reg_context {
    std::vector<ggml_backend_device>                 devices;
    std::vector<ggml_backend_tilelang_device_context> dev_ctxs;
};

static const char * ggml_backend_tilelang_reg_get_name(ggml_backend_reg_t reg) {
    GGML_UNUSED(reg);
    return GGML_TILELANG_NAME;
}

static size_t ggml_backend_tilelang_reg_get_device_count(ggml_backend_reg_t reg) {
    auto * rctx = (ggml_backend_tilelang_reg_context *) reg->context;
    return rctx->devices.size();
}

static ggml_backend_dev_t ggml_backend_tilelang_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    auto * rctx = (ggml_backend_tilelang_reg_context *) reg->context;
    GGML_ASSERT(index < rctx->devices.size());
    return &rctx->devices[index];
}

static void * ggml_backend_tilelang_reg_get_proc_address(ggml_backend_reg_t reg, const char * name) {
    GGML_UNUSED(reg);
    GGML_UNUSED(name);
    return nullptr;
}

static const struct ggml_backend_reg_i ggml_backend_tilelang_reg_iface = {
    /* .get_name         = */ ggml_backend_tilelang_reg_get_name,
    /* .get_device_count = */ ggml_backend_tilelang_reg_get_device_count,
    /* .get_device       = */ ggml_backend_tilelang_reg_get_device,
    /* .get_proc_address = */ ggml_backend_tilelang_reg_get_proc_address,
};

ggml_backend_reg_t ggml_backend_tilelang_reg(void) {
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);

    static ggml_backend_reg                    s_reg = {};
    static ggml_backend_tilelang_reg_context * s_rctx = nullptr;
    static bool                                s_initialized = false;

    if (!s_initialized) {
        s_rctx = new ggml_backend_tilelang_reg_context();

        const int n = ggml_backend_tilelang_get_device_count_impl();
        s_rctx->devices.resize(n);
        s_rctx->dev_ctxs.resize(n);

        s_reg.api_version = GGML_BACKEND_API_VERSION;
        s_reg.iface       = ggml_backend_tilelang_reg_iface;
        s_reg.context     = s_rctx;

        for (int i = 0; i < n; ++i) {
            s_rctx->dev_ctxs[i].device = i;
            s_rctx->dev_ctxs[i].name   = std::string(GGML_TILELANG_NAME) + std::to_string(i);

            cudaDeviceProp prop = {};
            cudaError_t    err  = cudaGetDeviceProperties(&prop, i);
            if (err == cudaSuccess) {
                s_rctx->dev_ctxs[i].description = prop.name;
            } else {
                (void) cudaGetLastError();
                s_rctx->dev_ctxs[i].description = "Unknown CUDA device";
            }

            s_rctx->devices[i] = {
                /* .iface   = */ ggml_backend_tilelang_device_iface,
                /* .reg     = */ &s_reg,
                /* .context = */ &s_rctx->dev_ctxs[i],
            };
        }
        s_initialized = true;
    }

    return &s_reg;
}

// ---------------------------------------------------------------------------
// public API
// ---------------------------------------------------------------------------

ggml_backend_t ggml_backend_tilelang_init(int device) {
    if (device < 0 || device >= ggml_backend_tilelang_get_device_count_impl()) {
        GGML_TILELANG_LOG_ERROR("[tilelang] init: invalid device %d\n", device);
        return nullptr;
    }

    auto * ctx = new ggml_backend_tilelang_context(device);

    ggml_backend_t backend = new ggml_backend {
        /* .guid    = */ ggml_backend_tilelang_guid(),
        /* .iface   = */ ggml_backend_tilelang_iface,
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_tilelang_reg(), device),
        /* .context = */ ctx,
    };
    return backend;
}

int ggml_backend_tilelang_get_device_count(void) {
    return ggml_backend_tilelang_get_device_count_impl();
}

void ggml_backend_tilelang_get_device_description(int device, char * description, size_t description_size) {
    if (description == nullptr || description_size == 0) {
        return;
    }
    description[0] = '\0';
    if (device < 0 || device >= ggml_backend_tilelang_get_device_count_impl()) {
        return;
    }
    cudaDeviceProp prop = {};
    cudaError_t err = cudaGetDeviceProperties(&prop, device);
    if (err == cudaSuccess) {
        std::snprintf(description, description_size, "%s", prop.name);
    } else {
        (void) cudaGetLastError();
    }
}

void ggml_backend_tilelang_get_device_memory(int device, size_t * free, size_t * total) {
    if (free)  *free  = 0;
    if (total) *total = 0;
    if (device < 0 || device >= ggml_backend_tilelang_get_device_count_impl()) {
        return;
    }
    ggml_backend_tilelang_set_device(device);
    cudaError_t err = cudaMemGetInfo(free, total);
    if (err != cudaSuccess) {
        (void) cudaGetLastError();
        if (free)  *free  = 0;
        if (total) *total = 0;
    }
}

ggml_backend_buffer_type_t ggml_backend_tilelang_buffer_type(int device) {
    return ggml_backend_tilelang_buffer_type_impl(device);
}

// ---------------------------------------------------------------------------
// dynamic-loader entry points
// ---------------------------------------------------------------------------

static int ggml_backend_tilelang_score(void) {
    int n = ggml_backend_tilelang_get_device_count_impl();
    if (n <= 0) {
        return 0;
    }
    return 50; // arbitrary positive score; lower than CUDA so it doesn't compete by default
}

GGML_BACKEND_DL_IMPL(ggml_backend_tilelang_reg)
GGML_BACKEND_DL_SCORE_IMPL(ggml_backend_tilelang_score)
