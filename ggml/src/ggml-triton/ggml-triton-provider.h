#pragma once

#include "ggml.h"

#include <vector>
#include <unordered_map>
#include <string>

// Forward declaration
struct ggml_backend_triton_context;

// Provider types
enum ggml_triton_provider_type {
    GGML_TRITON_PROVIDER_TRITON   = 0,  // Triton AOT kernels
    GGML_TRITON_PROVIDER_CUTLASS  = 1,  // CUTLASS compiled kernels
    GGML_TRITON_PROVIDER_TILELANG = 2,  // TileLang compiled kernels
    GGML_TRITON_PROVIDER_AUTO     = 3,  // Auto-select best
    GGML_TRITON_PROVIDER_CPU      = 4,  // Pure CPU reference implementation
};

// A single kernel implementation for an op
struct ggml_triton_kernel_impl {
    const char * name;                   // Human-readable name, e.g. "cutlass_gemm_f16_sm80"
    ggml_triton_provider_type provider;  // Which provider this comes from

    // Check if this implementation supports the given tensor operation
    // (checks dtype, shape, layout constraints etc.)
    bool (* supports)(const struct ggml_tensor * op);

    // Execute the kernel for this tensor operation
    // Returns true on success, false on failure
    bool (* execute)(struct ggml_backend_triton_context * ctx, const struct ggml_tensor * node);

    int priority;  // Higher value = preferred when multiple impls support the same op
};

// Registry of kernel implementations
class ggml_triton_op_registry {
public:
    // Register a kernel implementation for a specific op
    void register_impl(enum ggml_op op, const ggml_triton_kernel_impl & impl);

    // Select the best implementation for a given tensor operation
    // Returns nullptr if no implementation supports this op
    const ggml_triton_kernel_impl * select(const struct ggml_tensor * op) const;

    // Check if any implementation supports this op
    bool has_impl(const struct ggml_tensor * op) const;

    // Get all registered implementations for an op (for debugging/logging)
    const std::vector<ggml_triton_kernel_impl> * get_impls(enum ggml_op op) const;

private:
    std::unordered_map<int, std::vector<ggml_triton_kernel_impl>> impls_;
};

// Get the global op registry (lazily initialized with builtin providers)
ggml_triton_op_registry & ggml_triton_global_registry();

// Initialize and register all built-in Triton providers
void ggml_triton_register_builtin_providers(ggml_triton_op_registry & registry);

// Initialize and register CPU reference providers
void ggml_triton_register_cpu_providers(ggml_triton_op_registry & registry);
