#include "ggml-triton-provider.h"

#ifdef GGML_TRITON_HAS_RMSNORM
#include "ggml-triton-provider-rmsnorm.h"
#endif

#ifdef GGML_TRITON_HAS_ROPE
#include "ggml-triton-provider-rope.h"
#endif

#ifdef GGML_TRITON_HAS_FLASH_ATTN
#include "ggml-triton-provider-flash-attn.h"
#endif

#ifdef GGML_TRITON_HAS_CUTLASS
#include "ggml-triton-provider-cutlass.h"
#endif

#ifdef GGML_TRITON_HAS_TILELANG
#include "ggml-triton-provider-tilelang.h"
#endif

#include <mutex>

// ----------------------------------------------------------------------------
// ggml_triton_op_registry implementation
// ----------------------------------------------------------------------------

void ggml_triton_op_registry::register_impl(enum ggml_op op, const ggml_triton_kernel_impl & impl) {
    impls_[static_cast<int>(op)].push_back(impl);
}

const ggml_triton_kernel_impl * ggml_triton_op_registry::select(const struct ggml_tensor * op) const {
    if (op == nullptr) {
        return nullptr;
    }

    auto it = impls_.find(static_cast<int>(op->op));
    if (it == impls_.end()) {
        return nullptr;
    }

    const ggml_triton_kernel_impl * best = nullptr;
    for (const auto & impl : it->second) {
        if (impl.supports != nullptr && impl.supports(op)) {
            if (best == nullptr || impl.priority > best->priority) {
                best = &impl;
            }
        }
    }
    return best;
}

bool ggml_triton_op_registry::has_impl(const struct ggml_tensor * op) const {
    return select(op) != nullptr;
}

const std::vector<ggml_triton_kernel_impl> * ggml_triton_op_registry::get_impls(enum ggml_op op) const {
    auto it = impls_.find(static_cast<int>(op));
    if (it == impls_.end()) {
        return nullptr;
    }
    return &it->second;
}

// ----------------------------------------------------------------------------
// Global registry singleton
// ----------------------------------------------------------------------------

ggml_triton_op_registry & ggml_triton_global_registry() {
    static ggml_triton_op_registry registry;
    static std::once_flag flag;
    std::call_once(flag, []() {
        ggml_triton_register_cpu_providers(registry);
        ggml_triton_register_builtin_providers(registry);
#ifdef GGML_TRITON_HAS_RMSNORM
        ggml_triton_register_rmsnorm_providers(registry);
#endif
#ifdef GGML_TRITON_HAS_ROPE
        ggml_triton_register_rope_providers(registry);
#endif
#ifdef GGML_TRITON_HAS_FLASH_ATTN
        ggml_triton_register_flash_attn_providers(registry);
#endif
#ifdef GGML_TRITON_HAS_CUTLASS
        ggml_triton_register_cutlass_providers(registry);
#endif

#ifdef GGML_TRITON_HAS_TILELANG
        ggml_triton_register_tilelang_providers(registry);
#endif
    });
    return registry;
}
