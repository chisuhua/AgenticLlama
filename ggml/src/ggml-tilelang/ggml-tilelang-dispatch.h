#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#include <cuda_runtime.h>

// Returns true if this backend can compute the given op tensor.
// Initial scope: GGML_OP_ADD and GGML_OP_MUL on F16 / F32, contiguous, equal-shape.
bool ggml_backend_tilelang_supports_op(const struct ggml_tensor * op);

// Dispatch a single ggml node to the appropriate TileLang AOT kernel.
// Returns GGML_STATUS_SUCCESS on success.
// Pre-conditions: ggml_backend_tilelang_supports_op(node) returned true and the
// device context's stream has been made current.
enum ggml_status ggml_backend_tilelang_dispatch(struct ggml_tensor * node, cudaStream_t stream);
