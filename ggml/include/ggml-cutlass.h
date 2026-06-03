#pragma once

#include "ggml-backend.h"

#ifdef __cplusplus
extern "C" {
#endif

// Backend registry entry point. Returns the singleton ggml_backend_reg_t for
// the CUTLASS backend. Used by ggml-backend-reg.cpp to register the backend
// statically when GGML_USE_CUTLASS is defined, and discovered by name
// `ggml_backend_init` when loaded dynamically (GGML_BACKEND_DL).
GGML_BACKEND_API ggml_backend_reg_t ggml_backend_cutlass_reg(void);

// Convenience initializer: equivalent to
//     ggml_backend_dev_init(ggml_backend_dev_get_by_type(GPU), nullptr)
// but specifically returns a CUTLASS backend instance for the given device id.
GGML_BACKEND_API ggml_backend_t ggml_backend_cutlass_init(int device);

// Returns the buffer type used to allocate device memory for a particular
// CUTLASS device id.
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_cutlass_buffer_type(int device);

// Returns true if the given backend handle is a CUTLASS backend.
GGML_BACKEND_API bool ggml_backend_is_cutlass(ggml_backend_t backend);

// Total number of CUTLASS-capable devices visible to this process. Returns 0
// when no CUDA-capable GPU is present.
GGML_BACKEND_API int ggml_backend_cutlass_get_device_count(void);

#ifdef __cplusplus
}
#endif
