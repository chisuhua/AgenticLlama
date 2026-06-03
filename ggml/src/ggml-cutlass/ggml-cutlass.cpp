// =============================================================================
// CUTLASS backend - top-level registry & backend implementation.
// =============================================================================

#include <atomic>
#include <cstdio>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "ggml.h"
#include "ggml-impl.h"
#include "ggml-backend.h"
#include "ggml-backend-impl.h"

#include "ggml-cutlass.h"
#include "ggml-cutlass-context.h"
#include "ggml-cutlass-dispatch.h"

// =============================================================================
// Helpers (declared in ggml-cutlass-context.h)
// =============================================================================

namespace ggml_cutlass {

static std::atomic<int>  g_device_count{-1};
static thread_local int  g_active_device = -1;

int device_count() {
    int cached = g_device_count.load(std::memory_order_acquire);
    if (cached >= 0) {
        return cached;
    }

    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess) {
        // Clear sticky error and treat as "no devices".
        (void) cudaGetLastError();
        count = 0;
    }
    if (count > GGML_CUTLASS_MAX_DEVICES) {
        count = GGML_CUTLASS_MAX_DEVICES;
    }
    g_device_count.store(count, std::memory_order_release);
    return count;
}

void set_device(int device) {
    if (device == g_active_device) {
        return;
    }
    cudaError_t err = cudaSetDevice(device);
    if (err != cudaSuccess) {
        GGML_LOG_ERROR("CUTLASS backend: cudaSetDevice(%d) failed: %s\n",
                       device, cudaGetErrorString(err));
        return;
    }
    g_active_device = device;
}

bool query_device_info(int device, ggml_cutlass_device_info & info) {
    if (device < 0 || device >= device_count()) {
        return false;
    }

    cudaDeviceProp prop{};
    cudaError_t err = cudaGetDeviceProperties(&prop, device);
    if (err != cudaSuccess) {
        (void) cudaGetLastError();
        return false;
    }

    info.device_id     = device;
    info.compute_major = prop.major;
    info.compute_minor = prop.minor;
    info.sm_count      = prop.multiProcessorCount;
    info.warp_size     = prop.warpSize;
    info.total_vram    = prop.totalGlobalMem;
    info.name          = prop.name;

    char bus_id[32];
    err = cudaDeviceGetPCIBusId(bus_id, sizeof(bus_id), device);
    if (err == cudaSuccess) {
        info.pci_bus_id = bus_id;
        // Lower-case to match the convention used elsewhere in ggml.
        for (auto & c : info.pci_bus_id) {
            if (c >= 'A' && c <= 'Z') c = char(c - 'A' + 'a');
        }
    } else {
        (void) cudaGetLastError();
        info.pci_bus_id.clear();
    }
    return true;
}

} // namespace ggml_cutlass

// =============================================================================
// ggml_backend_cutlass_context lifecycle
// =============================================================================

ggml_backend_cutlass_context::~ggml_backend_cutlass_context() {
    if (workspace != nullptr) {
        ggml_cutlass::set_device(device_id);
        cudaError_t err = cudaFree(workspace);
        if (err != cudaSuccess) {
            GGML_LOG_ERROR("CUTLASS backend: workspace cudaFree failed: %s\n",
                           cudaGetErrorString(err));
        }
        workspace      = nullptr;
        workspace_size = 0;
    }
    if (owns_stream && stream != nullptr) {
        ggml_cutlass::set_device(device_id);
        cudaError_t err = cudaStreamDestroy(stream);
        if (err != cudaSuccess) {
            GGML_LOG_ERROR("CUTLASS backend: cudaStreamDestroy failed: %s\n",
                           cudaGetErrorString(err));
        }
    }
    stream = nullptr;
}

cudaError_t ggml_backend_cutlass_context::ensure_workspace(size_t size) {
    if (size <= workspace_size) {
        return cudaSuccess;
    }
    ggml_cutlass::set_device(device_id);
    if (workspace != nullptr) {
        cudaFree(workspace);
        workspace = nullptr;
        workspace_size = 0;
    }
    cudaError_t err = cudaMalloc(&workspace, size);
    if (err != cudaSuccess) {
        (void) cudaGetLastError();
        workspace = nullptr;
        workspace_size = 0;
        return err;
    }
    workspace_size = size;
    return cudaSuccess;
}

// =============================================================================
// ggml_backend_i implementation
// =============================================================================

static ggml_guid_t ggml_backend_cutlass_guid(void) {
    // Random GUID; just needs to be unique among ggml backends.
    static ggml_guid guid = { 0x7c, 0x2a, 0xe3, 0x91, 0x44, 0x68, 0x4f, 0xa1,
                              0x9b, 0x05, 0xd1, 0x33, 0x6e, 0x82, 0x57, 0xc4 };
    return &guid;
}

static const char * ggml_backend_cutlass_get_name(ggml_backend_t backend) {
    auto * ctx = (ggml_backend_cutlass_context *) backend->context;
    return ctx->name.c_str();
}

static void ggml_backend_cutlass_free(ggml_backend_t backend) {
    auto * ctx = (ggml_backend_cutlass_context *) backend->context;
    delete ctx;
    delete backend;
}

static void ggml_backend_cutlass_synchronize(ggml_backend_t backend) {
    auto * ctx = (ggml_backend_cutlass_context *) backend->context;
    ggml_cutlass::set_device(ctx->device_id);
    GGML_CUTLASS_CHECK(cudaStreamSynchronize(ctx->stream));
}

static void ggml_backend_cutlass_set_tensor_async(
        ggml_backend_t backend, ggml_tensor * tensor,
        const void * data, size_t offset, size_t size) {
    auto * ctx = (ggml_backend_cutlass_context *) backend->context;
    GGML_ASSERT(ggml_backend_buffer_is_cutlass(tensor->buffer));
    ggml_cutlass::set_device(ctx->device_id);
    GGML_CUTLASS_CHECK(cudaMemcpyAsync(
        (char *) tensor->data + offset, data, size,
        cudaMemcpyHostToDevice, ctx->stream));
}

static void ggml_backend_cutlass_get_tensor_async(
        ggml_backend_t backend, const ggml_tensor * tensor,
        void * data, size_t offset, size_t size) {
    auto * ctx = (ggml_backend_cutlass_context *) backend->context;
    GGML_ASSERT(ggml_backend_buffer_is_cutlass(tensor->buffer));
    ggml_cutlass::set_device(ctx->device_id);
    GGML_CUTLASS_CHECK(cudaMemcpyAsync(
        data, (const char *) tensor->data + offset, size,
        cudaMemcpyDeviceToHost, ctx->stream));
}

static enum ggml_status ggml_backend_cutlass_graph_compute(
        ggml_backend_t backend, ggml_cgraph * cgraph) {
    auto * ctx = (ggml_backend_cutlass_context *) backend->context;
    ggml_cutlass::set_device(ctx->device_id);

    for (int i = 0; i < cgraph->n_nodes; ++i) {
        ggml_tensor * node = cgraph->nodes[i];

        if ((node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            continue;
        }

        ggml_status st = ggml_cutlass::compute_op(*ctx, node);
        if (st != GGML_STATUS_SUCCESS) {
            return st;
        }
    }
    return GGML_STATUS_SUCCESS;
}

static const ggml_backend_i ggml_backend_cutlass_interface = {
    /* .get_name              = */ ggml_backend_cutlass_get_name,
    /* .free                  = */ ggml_backend_cutlass_free,
    /* .set_tensor_async      = */ ggml_backend_cutlass_set_tensor_async,
    /* .get_tensor_async      = */ ggml_backend_cutlass_get_tensor_async,
    /* .set_tensor_2d_async   = */ NULL,
    /* .get_tensor_2d_async   = */ NULL,
    /* .cpy_tensor_async      = */ NULL,
    /* .synchronize           = */ ggml_backend_cutlass_synchronize,
    /* .graph_plan_create     = */ NULL,
    /* .graph_plan_free       = */ NULL,
    /* .graph_plan_update     = */ NULL,
    /* .graph_plan_compute    = */ NULL,
    /* .graph_compute         = */ ggml_backend_cutlass_graph_compute,
    /* .event_record          = */ NULL,
    /* .event_wait            = */ NULL,
    /* .graph_optimize        = */ NULL,
};

bool ggml_backend_is_cutlass(ggml_backend_t backend) {
    return backend != nullptr &&
           ggml_guid_matches(backend->guid, ggml_backend_cutlass_guid());
}

int ggml_backend_cutlass_get_device_count(void) {
    return ggml_cutlass::device_count();
}

// =============================================================================
// ggml_backend_device_i implementation
// =============================================================================

static const char * ggml_backend_cutlass_device_get_name(ggml_backend_dev_t dev) {
    auto * ctx = (ggml_backend_cutlass_device_context *) dev->context;
    return ctx->name.c_str();
}

static const char * ggml_backend_cutlass_device_get_description(ggml_backend_dev_t dev) {
    auto * ctx = (ggml_backend_cutlass_device_context *) dev->context;
    return ctx->description.c_str();
}

static void ggml_backend_cutlass_device_get_memory(
        ggml_backend_dev_t dev, size_t * free, size_t * total) {
    auto * ctx = (ggml_backend_cutlass_device_context *) dev->context;
    ggml_cutlass::set_device(ctx->device);
    cudaError_t err = cudaMemGetInfo(free, total);
    if (err != cudaSuccess) {
        (void) cudaGetLastError();
        *free  = 0;
        *total = 0;
    }
}

static enum ggml_backend_dev_type ggml_backend_cutlass_device_get_type(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    return GGML_BACKEND_DEVICE_TYPE_GPU;
}

static void ggml_backend_cutlass_device_get_props(
        ggml_backend_dev_t dev, struct ggml_backend_dev_props * props) {
    auto * ctx = (ggml_backend_cutlass_device_context *) dev->context;

    props->name        = ggml_backend_cutlass_device_get_name(dev);
    props->description = ggml_backend_cutlass_device_get_description(dev);
    props->type        = ggml_backend_cutlass_device_get_type(dev);
    props->device_id   = ctx->pci_bus_id.empty() ? nullptr : ctx->pci_bus_id.c_str();
    ggml_backend_cutlass_device_get_memory(dev, &props->memory_free, &props->memory_total);

    props->caps = {
        /* .async                = */ true,
        /* .host_buffer          = */ false,  // pinned host buffers not implemented yet
        /* .buffer_from_host_ptr = */ false,
        /* .events               = */ false,
    };
}

ggml_backend_t ggml_backend_cutlass_init(int device) {
    if (device < 0 || device >= ggml_cutlass::device_count()) {
        GGML_LOG_ERROR("CUTLASS backend: invalid device id %d\n", device);
        return nullptr;
    }

    ggml_cutlass::set_device(device);

    auto ctx = std::make_unique<ggml_backend_cutlass_context>();
    ctx->device_id = device;
    if (!ggml_cutlass::query_device_info(device, ctx->info)) {
        GGML_LOG_ERROR("CUTLASS backend: failed to query device %d\n", device);
        return nullptr;
    }

    cudaError_t err = cudaStreamCreateWithFlags(&ctx->stream, cudaStreamNonBlocking);
    if (err != cudaSuccess) {
        (void) cudaGetLastError();
        GGML_LOG_ERROR("CUTLASS backend: cudaStreamCreate failed: %s\n", cudaGetErrorString(err));
        return nullptr;
    }
    ctx->owns_stream = true;

    char name[64];
    snprintf(name, sizeof(name), "CUTLASS%d", device);
    ctx->name = name;

    ggml_backend_dev_t dev = nullptr;
    ggml_backend_reg_t reg = ggml_backend_cutlass_reg();
    if (reg != nullptr) {
        for (size_t i = 0; i < ggml_backend_reg_dev_count(reg); ++i) {
            ggml_backend_dev_t cand = ggml_backend_reg_dev_get(reg, i);
            auto * cctx = (ggml_backend_cutlass_device_context *) cand->context;
            if (cctx != nullptr && cctx->device == device) {
                dev = cand;
                break;
            }
        }
    }

    auto * backend = new ggml_backend{
        /* .guid    = */ ggml_backend_cutlass_guid(),
        /* .iface   = */ ggml_backend_cutlass_interface,
        /* .device  = */ dev,
        /* .context = */ ctx.release(),
    };
    return backend;
}

static ggml_backend_t ggml_backend_cutlass_device_init_backend(
        ggml_backend_dev_t dev, const char * params) {
    GGML_UNUSED(params);
    auto * ctx = (ggml_backend_cutlass_device_context *) dev->context;
    return ggml_backend_cutlass_init(ctx->device);
}

static ggml_backend_buffer_type_t ggml_backend_cutlass_device_get_buffer_type(ggml_backend_dev_t dev) {
    auto * ctx = (ggml_backend_cutlass_device_context *) dev->context;
    ggml_backend_buffer_type_t buft = ggml_backend_cutlass_buffer_type(ctx->device);
    if (buft != nullptr && buft->device == nullptr) {
        // Wire the buffer type back to its device on first use.
        buft->device = dev;
    }
    return buft;
}

static bool ggml_backend_cutlass_device_supports_op(
        ggml_backend_dev_t dev, const ggml_tensor * op) {
    GGML_UNUSED(dev);
    return ggml_cutlass::supports_op(op);
}

static bool ggml_backend_cutlass_device_supports_buft(
        ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    auto * ctx = (ggml_backend_cutlass_device_context *) dev->context;
    if (buft == nullptr || buft->iface.get_name == nullptr) {
        return false;
    }
    // Only accept our own buffer type for the matching device.
    if (buft->iface.get_name != ggml_backend_cutlass_buffer_type(ctx->device)->iface.get_name) {
        return false;
    }
    auto * buft_ctx = (ggml_backend_cutlass_buffer_type_context *) buft->context;
    return buft_ctx != nullptr && buft_ctx->device == ctx->device;
}

static const ggml_backend_device_i ggml_backend_cutlass_device_interface = {
    /* .get_name             = */ ggml_backend_cutlass_device_get_name,
    /* .get_description      = */ ggml_backend_cutlass_device_get_description,
    /* .get_memory           = */ ggml_backend_cutlass_device_get_memory,
    /* .get_type             = */ ggml_backend_cutlass_device_get_type,
    /* .get_props            = */ ggml_backend_cutlass_device_get_props,
    /* .init_backend         = */ ggml_backend_cutlass_device_init_backend,
    /* .get_buffer_type      = */ ggml_backend_cutlass_device_get_buffer_type,
    /* .get_host_buffer_type = */ NULL,
    /* .buffer_from_host_ptr = */ NULL,
    /* .supports_op          = */ ggml_backend_cutlass_device_supports_op,
    /* .supports_buft        = */ ggml_backend_cutlass_device_supports_buft,
    /* .offload_op           = */ NULL,
    /* .event_new            = */ NULL,
    /* .event_free           = */ NULL,
    /* .event_synchronize    = */ NULL,
};

// =============================================================================
// ggml_backend_reg_i implementation
// =============================================================================

struct ggml_backend_cutlass_reg_context {
    std::vector<ggml_backend_device>                          devices;
    std::vector<ggml_backend_cutlass_device_context>          contexts;
};

static const char * ggml_backend_cutlass_reg_get_name(ggml_backend_reg_t reg) {
    GGML_UNUSED(reg);
    return "CUTLASS";
}

static size_t ggml_backend_cutlass_reg_get_device_count(ggml_backend_reg_t reg) {
    auto * rctx = (ggml_backend_cutlass_reg_context *) reg->context;
    return rctx->devices.size();
}

static ggml_backend_dev_t ggml_backend_cutlass_reg_get_device(
        ggml_backend_reg_t reg, size_t index) {
    auto * rctx = (ggml_backend_cutlass_reg_context *) reg->context;
    GGML_ASSERT(index < rctx->devices.size());
    return &rctx->devices[index];
}

static void * ggml_backend_cutlass_reg_get_proc_address(
        ggml_backend_reg_t reg, const char * name) {
    GGML_UNUSED(reg);
    GGML_UNUSED(name);
    // No backend-specific procs are exposed yet.
    return NULL;
}

static const ggml_backend_reg_i ggml_backend_cutlass_reg_interface = {
    /* .get_name         = */ ggml_backend_cutlass_reg_get_name,
    /* .get_device_count = */ ggml_backend_cutlass_reg_get_device_count,
    /* .get_device       = */ ggml_backend_cutlass_reg_get_device,
    /* .get_proc_address = */ ggml_backend_cutlass_reg_get_proc_address,
};

ggml_backend_reg_t ggml_backend_cutlass_reg(void) {
    static std::mutex mutex;
    static std::unique_ptr<ggml_backend_cutlass_reg_context> rctx;
    static ggml_backend_reg reg_storage{};
    static bool initialized = false;

    std::lock_guard<std::mutex> lock(mutex);
    if (initialized) {
        return &reg_storage;
    }

    rctx = std::make_unique<ggml_backend_cutlass_reg_context>();

    const int n_dev = ggml_cutlass::device_count();
    rctx->contexts.resize(n_dev);
    rctx->devices.resize(n_dev);

    for (int i = 0; i < n_dev; ++i) {
        ggml_cutlass_device_info info;
        if (!ggml_cutlass::query_device_info(i, info)) {
            // Skip with a placeholder - keeping indices in sync with CUDA ids.
            info.device_id = i;
            info.name      = "Unknown CUDA device";
        }

        char short_name[32];
        snprintf(short_name, sizeof(short_name), "CUTLASS%d", i);

        rctx->contexts[i].device      = i;
        rctx->contexts[i].name        = short_name;
        rctx->contexts[i].description = info.name;
        rctx->contexts[i].pci_bus_id  = info.pci_bus_id;

        rctx->devices[i] = ggml_backend_device{
            /* .iface   = */ ggml_backend_cutlass_device_interface,
            /* .reg     = */ &reg_storage,
            /* .context = */ &rctx->contexts[i],
        };
    }

    reg_storage = ggml_backend_reg{
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ ggml_backend_cutlass_reg_interface,
        /* .context     = */ rctx.get(),
    };
    initialized = true;
    return &reg_storage;
}

// Optional score for dynamic-loading discovery. Returns 0 when CUDA isn't
// available so the backend won't be picked for systems without a GPU.
static int ggml_backend_cutlass_score(void) {
    return ggml_cutlass::device_count() > 0 ? 60 : 0;
}

GGML_BACKEND_DL_IMPL(ggml_backend_cutlass_reg)
GGML_BACKEND_DL_SCORE_IMPL(ggml_backend_cutlass_score)
