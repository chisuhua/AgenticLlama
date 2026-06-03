#pragma once

#include "ggml.h"
#include "ggml-backend.h"

struct ggml_backend_triton_context;

// Returns true if the Triton backend has a kernel for the given op.
// Uses the global provider registry so it can be called without a context
// (from the device interface before a backend instance exists).
bool ggml_backend_triton_supports_op(const struct ggml_tensor * op);

// Dispatch a single graph node via the provider registry.
// Returns GGML_STATUS_SUCCESS on success, GGML_STATUS_FAILED if the op is
// unsupported or the kernel launch failed.
enum ggml_status ggml_triton_dispatch_op(ggml_backend_triton_context * ctx,
                                         struct ggml_tensor * node);
