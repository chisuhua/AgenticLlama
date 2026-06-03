#include "ggml-triton-dispatch.h"
#include "ggml-triton-context.h"
#include "ggml-triton-provider.h"

#include "ggml-impl.h"

#include <cstdio>

// ----------------------------------------------------------------------------
// supports_op  (device-level, no context needed -- uses global registry)
// ----------------------------------------------------------------------------

bool ggml_backend_triton_supports_op(const struct ggml_tensor * op) {
    if (op == nullptr) {
        return false;
    }

    // Always allow no-op book-keeping ops so a graph can be assigned wholly to
    // the Triton backend during testing.
    switch (op->op) {
        case GGML_OP_NONE:
        case GGML_OP_VIEW:
        case GGML_OP_RESHAPE:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
        case GGML_OP_CONT:
            return true;
        default:
            break;
    }

    // Query the global provider registry
    return ggml_triton_global_registry().has_impl(op);
}

// ----------------------------------------------------------------------------
// dispatch  (backend-level, uses per-context registry)
// ----------------------------------------------------------------------------

enum ggml_status ggml_triton_dispatch_op(ggml_backend_triton_context * ctx,
                                         struct ggml_tensor * node) {
    GGML_ASSERT(ctx  != nullptr);
    GGML_ASSERT(node != nullptr);

    // No-op book-keeping ops
    switch (node->op) {
        case GGML_OP_NONE:
        case GGML_OP_VIEW:
        case GGML_OP_RESHAPE:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
        case GGML_OP_CONT:
            return GGML_STATUS_SUCCESS;
        default:
            break;
    }

    // Select and execute via the per-context registry
    const ggml_triton_kernel_impl * impl = ctx->op_registry.select(node);
    if (impl == nullptr) {
        GGML_LOG_ERROR("ggml-triton: unsupported op %s\n", ggml_op_name(node->op));
        return GGML_STATUS_FAILED;
    }

    bool ok = impl->execute(ctx, node);
    return ok ? GGML_STATUS_SUCCESS : GGML_STATUS_FAILED;
}
