# ggml-tilelang backend
#
# This backend ships TileLang AOT-compiled kernels (CUDA source files generated
# by `scripts/compile_kernels.py`) and links them into a regular CUDA library.
# Kernels are exposed as `extern "C"` launcher functions and called directly,
# i.e. there is no Driver API / cuLaunchKernel involved.

find_package(CUDAToolkit REQUIRED)
enable_language(CUDA)

# ----------------------------------------------------------------------------
# AOT compilation step (invokes TileLang to generate .cu files in `generated/`).
# In this skeleton the generated/ directory is checked-in with realistic mocks
# so the backend can be built without TileLang installed. When TileLang is
# available the user can re-run the AOT step manually:
#
#   python scripts/compile_kernels.py \
#       --config scripts/kernel_configs.json \
#       --out  ggml/src/ggml-tilelang/kernels/generated
# ----------------------------------------------------------------------------

set(TILELANG_GENERATED_DIR ${CMAKE_CURRENT_SOURCE_DIR}/kernels/generated)
set(TILELANG_INCLUDE_DIR   ${CMAKE_CURRENT_SOURCE_DIR}/kernels/include)

file(GLOB TILELANG_GENERATED_SOURCES CONFIGURE_DEPENDS
     "${TILELANG_GENERATED_DIR}/*.cu")

if (NOT TILELANG_GENERATED_SOURCES)
    message(WARNING "ggml-tilelang: no generated kernels found in ${TILELANG_GENERATED_DIR}, "
                    "the backend will be built but cannot launch any kernel.")
endif()

# ----------------------------------------------------------------------------
# Backend library
# ----------------------------------------------------------------------------

ggml_add_backend_library(ggml-tilelang
    ggml-tilelang.cpp
    ggml-tilelang-buffer.cpp
    ggml-tilelang-dispatch.cpp
    ${TILELANG_GENERATED_SOURCES}
)

target_include_directories(ggml-tilelang PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${TILELANG_INCLUDE_DIR}
)

target_link_libraries(ggml-tilelang PRIVATE CUDA::cudart)

set_target_properties(ggml-tilelang PROPERTIES
    CUDA_ARCHITECTURES "70;75;80;86;89;90"
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
)
