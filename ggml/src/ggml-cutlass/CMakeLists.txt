# =============================================================================
# CUTLASS backend for ggml.
#
# Enabled with -DGGML_CUTLASS=ON. Optionally uses generated CUTLASS GEMM
# instances via the CuteDSL Python codegen (-DGGML_CUTLASS_USE_DSL=ON).
# =============================================================================

option(GGML_CUTLASS         "ggml: enable CUTLASS backend"                          OFF)
option(GGML_CUTLASS_USE_DSL "ggml: use CuteDSL Python codegen for CUTLASS kernels"  OFF)

# This file is included by `ggml_add_backend(CUTLASS)` only when GGML_CUTLASS is
# ON, so we don't need to guard the whole file again. Keep one defensive check
# in case it is included directly.
if (NOT GGML_CUTLASS)
    return()
endif()

find_package(CUDAToolkit REQUIRED)
enable_language(CUDA)

set(CUTLASS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/cutlass" CACHE PATH
    "Path to CUTLASS source tree (header-only). Defaults to a git submodule under this directory.")

set(GGML_CUTLASS_SOURCES
    ggml-cutlass.cpp
    ggml-cutlass-buffer.cpp
    ggml-cutlass-dispatch.cpp
    kernels/gemm_f16.cu
    kernels/gemm_f32.cu
    kernels/gemm_q4_0.cu
    kernels/gemm_q8_0.cu
    kernels/elementwise.cu)

ggml_add_backend_library(ggml-cutlass ${GGML_CUTLASS_SOURCES})

target_include_directories(ggml-cutlass PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/kernels/include)

# Public include path (the public header lives under ggml/include).
# ggml_add_backend_library already adds .. so the public header is reachable
# via "ggml-cutlass.h".

# CUTLASS is header-only. If the headers are present, enable the CUTLASS path
# in gemm_f16.cu; otherwise fall back to the naive kernel.
if (EXISTS "${CUTLASS_DIR}/include/cutlass/cutlass.h")
    target_include_directories(ggml-cutlass PRIVATE
        "${CUTLASS_DIR}/include"
        "${CUTLASS_DIR}/tools/util/include")
    target_compile_definitions(ggml-cutlass PRIVATE GGML_CUTLASS_HAS_HEADERS)
    message(STATUS "CUTLASS backend: using headers from ${CUTLASS_DIR}")
else()
    message(WARNING
        "CUTLASS headers not found at ${CUTLASS_DIR}; "
        "falling back to naive CUDA kernels. "
        "Initialize the submodule (git submodule update --init "
        "${CUTLASS_DIR}) or set -DCUTLASS_DIR=<path> to enable the fast path.")
endif()

# Optional: drive a Python codegen step to generate CUTLASS GEMM instance code.
if (GGML_CUTLASS_USE_DSL)
    find_package(Python3 COMPONENTS Interpreter REQUIRED)
    set(GENERATED_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")
    file(MAKE_DIRECTORY "${GENERATED_DIR}")

    set(GEN_SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/scripts/generate_gemm_instances.py")
    set(GEN_OUTPUT "${GENERATED_DIR}/gemm_instances.inc")

    add_custom_command(
        OUTPUT  "${GEN_OUTPUT}"
        COMMAND "${Python3_EXECUTABLE}" "${GEN_SCRIPT}"
                --output "${GEN_OUTPUT}"
        DEPENDS "${GEN_SCRIPT}"
        COMMENT "Generating CUTLASS GEMM instances via CuteDSL"
        VERBATIM)

    add_custom_target(ggml-cutlass-gen DEPENDS "${GEN_OUTPUT}")
    add_dependencies(ggml-cutlass ggml-cutlass-gen)

    target_include_directories(ggml-cutlass PRIVATE "${GENERATED_DIR}")
    target_compile_definitions(ggml-cutlass PRIVATE GGML_CUTLASS_USE_DSL)
endif()

target_link_libraries(ggml-cutlass PRIVATE CUDA::cudart)

# Sensible default arch list covering Ampere through Hopper. Users can override
# via -DCMAKE_CUDA_ARCHITECTURES=... when configuring.
if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set_target_properties(ggml-cutlass PROPERTIES
        CUDA_ARCHITECTURES "70;75;80;86;89;90")
endif()

set_target_properties(ggml-cutlass PROPERTIES CUDA_STANDARD 17)
