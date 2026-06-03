#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#include "ggml-cutlass-context.h"

// =============================================================================
// Op dispatch surface for the CUTLASS backend.
//
// The dispatcher decides whether a given ggml_tensor (an op node) can be
// executed by this backend, and if so, runs the appropriate CUDA kernel.
// Keeping this layer separate from the registry plumbing keeps the supported-
// op table easy to extend.
// =============================================================================

namespace ggml_cutlass {

// Returns true if the backend can execute `op`. Mirrors the contract of
// ggml_backend_device_i::supports_op.
bool supports_op(const ggml_tensor * op);

// Execute a single op into its output tensor. Returns the resulting status;
// success on completion (kernel launched, errors propagated via cudaGetLastError).
ggml_status compute_op(ggml_backend_cutlass_context & ctx, ggml_tensor * node);

} // namespace ggml_cutlass

// Internal helper exported for the buffer translation unit.
bool ggml_backend_buffer_is_cutlass(ggml_backend_buffer_t buffer);
