#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#ifdef __cplusplus
extern "C" {
#endif

#define GGML_TILELANG_NAME      "TileLang"
#define GGML_TILELANG_MAX_DEVICES 16

// backend API
GGML_BACKEND_API ggml_backend_t ggml_backend_tilelang_init(int device);

GGML_BACKEND_API bool ggml_backend_is_tilelang(ggml_backend_t backend);

GGML_BACKEND_API int  ggml_backend_tilelang_get_device_count(void);
GGML_BACKEND_API void ggml_backend_tilelang_get_device_description(int device, char * description, size_t description_size);
GGML_BACKEND_API void ggml_backend_tilelang_get_device_memory(int device, size_t * free, size_t * total);

GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_tilelang_buffer_type(int device);

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_tilelang_reg(void);

#ifdef __cplusplus
}
#endif
