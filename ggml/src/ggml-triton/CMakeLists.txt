option(GGML_TRITON              "ggml: enable Triton AOT backend"            OFF)
option(GGML_TRITON_PRECOMPILED  "ggml: use precompiled Triton kernels (skip AOT step)" OFF)
option(GGML_TRITON_CPU_ONLY     "Build Triton backend with CPU-only providers (no CUDA)" OFF)

if (GGML_TRITON)
    if (NOT GGML_TRITON_CPU_ONLY)
        find_package(CUDAToolkit REQUIRED)
    else()
        message(STATUS "ggml-triton: CPU-only mode enabled (no CUDA required)")
    endif()

    # ------------------------------------------------------------------
    # Optional AOT compilation step.
    # ------------------------------------------------------------------
    set(TRITON_GENERATED_DIR ${CMAKE_CURRENT_SOURCE_DIR}/kernels/generated)
    set(TRITON_KERNELS_DIR   ${CMAKE_CURRENT_SOURCE_DIR}/../../../triton_kernels)
    set(TRITON_SCRIPTS_DIR   ${CMAKE_CURRENT_SOURCE_DIR}/../../../scripts)

    if (NOT GGML_TRITON_PRECOMPILED)
        find_package(Python3 COMPONENTS Interpreter)
        if (Python3_FOUND)
            add_custom_target(ggml-triton-aot
                COMMAND ${Python3_EXECUTABLE}
                        ${TRITON_SCRIPTS_DIR}/compile_kernels.py
                        --registry ${TRITON_SCRIPTS_DIR}/kernel_registry.json
                        --kernels  ${TRITON_KERNELS_DIR}
                        --out      ${TRITON_GENERATED_DIR}
                WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                COMMENT "Triton AOT: compiling kernels via compile_kernels.py"
                VERBATIM)
        else()
            message(WARNING "ggml-triton: Python3 not found; AOT step skipped, "
                            "checked-in mock kernels will be used")
        endif()
    endif()

    if (NOT GGML_TRITON_CPU_ONLY)
        set(GGML_TRITON_GENERATED_SRC
            ${TRITON_GENERATED_DIR}/gelu_fp16_sm80.c
            ${TRITON_GENERATED_DIR}/gelu_fp32_sm80.c
            ${TRITON_GENERATED_DIR}/silu_fp16_sm80.c
            ${TRITON_GENERATED_DIR}/silu_fp32_sm80.c
        )
        set(GGML_TRITON_GPU_SRC
            ggml-triton-provider-triton.cpp
            ${GGML_TRITON_GENERATED_SRC}
        )
    endif()

    ggml_add_backend_library(ggml-triton
                             ggml-triton.cpp
                             ggml-triton-buffer.cpp
                             ggml-triton-dispatch.cpp
                             ggml-triton-provider.cpp
                             ggml-triton-provider-cpu.cpp
                             ${GGML_TRITON_GPU_SRC}
                            )

    target_include_directories(ggml-triton PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/kernels/include
        ${CMAKE_CURRENT_SOURCE_DIR}/kernels/generated
    )

    if (NOT GGML_TRITON_CPU_ONLY)
        target_link_libraries(ggml-triton PRIVATE CUDA::cuda_driver)
    else()
        target_compile_definitions(ggml-triton PRIVATE GGML_TRITON_CPU_ONLY)
    endif()

    if (TARGET ggml-triton-aot AND NOT GGML_TRITON_PRECOMPILED)
        add_dependencies(ggml-triton ggml-triton-aot)
    endif()

    # ------------------------------------------------------------------
    # Optional: CUTLASS kernel provider
    # ------------------------------------------------------------------
    if (NOT GGML_TRITON_CPU_ONLY)
        option(GGML_TRITON_WITH_CUTLASS "Enable CUTLASS kernels in Triton backend" OFF)
    endif()

    if(GGML_TRITON_WITH_CUTLASS)
        enable_language(CUDA)

        set(GGML_CUTLASS_KERNELS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/kernels/cutlass)
        # CUTLASS_DIR removed — CUTLASS headers come from the kernels/cutlass/include/ dir
        # (the old `third_party/cutlass` subdir is not copied over; if CUTLASS headers are
        #  needed at build time they should be available via the existing kernels/cutlass
        #  include path that target_include_directories() below sets up)

        target_sources(ggml-triton PRIVATE
            ggml-triton-provider-cutlass.cpp
            ${GGML_CUTLASS_KERNELS_DIR}/gemm_f16.cu
            ${GGML_CUTLASS_KERNELS_DIR}/gemm_f32.cu
            ${GGML_CUTLASS_KERNELS_DIR}/gemm_q4_0.cu
            ${GGML_CUTLASS_KERNELS_DIR}/gemm_q8_0.cu
            ${GGML_CUTLASS_KERNELS_DIR}/elementwise.cu
        )

        target_include_directories(ggml-triton PRIVATE
            ${GGML_CUTLASS_KERNELS_DIR}/include
        )

        # Need CUDA Runtime API for CUTLASS kernels
        target_link_libraries(ggml-triton PRIVATE CUDA::cudart)
        target_compile_definitions(ggml-triton PRIVATE GGML_TRITON_HAS_CUTLASS)

        # Set CUDA architectures for .cu files
        set_source_files_properties(
            ${GGML_CUTLASS_KERNELS_DIR}/gemm_f16.cu
            ${GGML_CUTLASS_KERNELS_DIR}/gemm_f32.cu
            ${GGML_CUTLASS_KERNELS_DIR}/gemm_q4_0.cu
            ${GGML_CUTLASS_KERNELS_DIR}/gemm_q8_0.cu
            ${GGML_CUTLASS_KERNELS_DIR}/elementwise.cu
            PROPERTIES LANGUAGE CUDA
        )
    endif()

    # ------------------------------------------------------------------
    # Optional: TileLang kernel provider
    # ------------------------------------------------------------------
    if (NOT GGML_TRITON_CPU_ONLY)
        option(GGML_TRITON_WITH_TILELANG "Enable TileLang kernels in Triton backend" OFF)
    endif()

    if(GGML_TRITON_WITH_TILELANG)
        set(GGML_TILELANG_KERNELS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/kernels/tilelang)
        set(GGML_TILELANG_GENERATED_DIR ${GGML_TILELANG_KERNELS_DIR}/generated)
        set(GGML_TILELANG_INCLUDE_DIR   ${GGML_TILELANG_KERNELS_DIR}/include)

        file(GLOB GGML_TILELANG_GENERATED_SOURCES CONFIGURE_DEPENDS
             "${GGML_TILELANG_GENERATED_DIR}/*.cu")

        if (NOT GGML_TILELANG_GENERATED_SOURCES)
            message(WARNING "ggml-triton: no TileLang generated kernels in ${GGML_TILELANG_GENERATED_DIR}; "
                            "provider will be registered but cannot launch any kernel.")
        endif()

        target_sources(ggml-triton PRIVATE
            ggml-triton-provider-tilelang.cpp
            ${GGML_TILELANG_GENERATED_SOURCES}
        )

        target_include_directories(ggml-triton PRIVATE
            ${GGML_TILELANG_INCLUDE_DIR}
        )

        target_link_libraries(ggml-triton PRIVATE CUDA::cudart)
        target_compile_definitions(ggml-triton PRIVATE GGML_TRITON_HAS_TILELANG)

        set_source_files_properties(
            ${GGML_TILELANG_GENERATED_SOURCES}
            PROPERTIES LANGUAGE CUDA
        )
    endif()
endif()
