#pragma once

#include "ggml-backend.h"

#ifdef __cplusplus
extern "C" {
#endif

// Backend registration entry point for the Triton AOT backend.
GGML_BACKEND_API ggml_backend_reg_t ggml_backend_triton_reg(void);

// Initialize a Triton backend on the given device index.
GGML_BACKEND_API ggml_backend_t ggml_backend_triton_init(int device);

// Number of Triton-capable devices visible to the process.
GGML_BACKEND_API int ggml_backend_triton_get_device_count(void);

#ifdef __cplusplus
}
#endif
