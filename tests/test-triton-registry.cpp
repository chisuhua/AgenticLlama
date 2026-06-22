// tests/test-triton-registry.cpp
//
// Unit test for the ggml-triton provider registry.
//
// Verifies that the built-in providers (CPU, Triton AOT, CUTLASS, TileLang)
// are registered correctly and that the registry's `select()` returns a
// non-null implementation for ops that have at least one provider.
//
// This test is gated on GGML_TRITON being enabled. Some providers (CUTLASS,
// TileLang) are only registered when their respective CMake options are on.
// The test gracefully skips assertions for providers that aren't built.

#include "ggml.h"
#include "ggml-triton-provider.h"
#include "ggml-triton-dispatch.h"
#include "ggml-impl.h"  // for ggml_set_op_params_i32 (internal API)

#include <cassert>
#include <cstdio>
#include <cstdlib>

int main() {
    // 1. Registry is non-empty: at least the CPU providers are always there.
    auto & reg = ggml_triton_global_registry();

    bool found_cpu_gelu = false;
    if (auto * impls = reg.get_impls(GGML_OP_UNARY)) {
        for (const auto & impl : *impls) {
            if (impl.provider == GGML_TRITON_PROVIDER_CPU &&
                std::string(impl.name).find("gelu") != std::string::npos) {
                found_cpu_gelu = true;
                break;
            }
        }
    }
    if (!found_cpu_gelu) {
        std::fprintf(stderr, "CPU GELU provider not registered (should always be present)\n");
        return 1;
    }

    // 2. Selecting the highest-priority provider for a valid f32 GELU
    //    returns a non-null impl.
    ggml_tensor a{};
    a.type = GGML_TYPE_F32;
    ggml_tensor unary_op{};
    unary_op.op   = GGML_OP_UNARY;
    unary_op.type = GGML_TYPE_F32;
    // Set the unary op type to GELU via the op_params I32 slot.
    ggml_set_op_params_i32(&unary_op, 0, GGML_UNARY_OP_GELU);
    unary_op.src[0] = &a;

    const auto * impl = reg.select(&unary_op);
    if (impl == nullptr) {
        std::fprintf(stderr, "No provider selected for f32 GELU\n");
        return 2;
    }

    std::printf("selected=%s provider=%d priority=%d\n",
                impl->name, (int) impl->provider, impl->priority);

    // 3. If TileLang provider is built (GGML_TRITON_HAS_TILELANG), it must
    //    be present in the registry for GGML_OP_ADD. We don't know at
    //    compile time whether the macro is defined (it's a CMake-injected
    //    define), so we check by name instead.
    bool found_tilelang = false;
    if (auto * impls = reg.get_impls(GGML_OP_ADD)) {
        for (const auto & impl : *impls) {
            if (impl.provider == GGML_TRITON_PROVIDER_TILELANG) {
                found_tilelang = true;
                break;
            }
        }
    }

#ifdef GGML_TRITON_HAS_TILELANG
    if (!found_tilelang) {
        std::fprintf(stderr, "TileLang ADD provider not registered (expected when GGML_TRITON_HAS_TILELANG is defined)\n");
        return 3;
    }
    std::printf("tilelang provider: registered (as expected)\n");
#else
    std::printf("tilelang provider: %s (skipping assertion — not built)\n",
                found_tilelang ? "registered" : "not registered");
#endif

    // Assert 4 (B.1): the Triton AOT RMSNorm provider (fp16 + fp32) must be
    // registered for GGML_OP_RMS_NORM. The CPU fp32 entry already exists
    // (see ggml-triton-provider-cpu.cpp:785-790) — we are specifically
    // asserting that the *triton AOT* entries get added by the new
    // ggml-triton-provider-rmsnorm.{h,cpp} files (Task 8) and are reachable
    // from the global registry.
    bool found_triton_rms_norm_fp16 = false;
    bool found_triton_rms_norm_fp32 = false;
    if (auto * impls = reg.get_impls(GGML_OP_RMS_NORM)) {
        for (const auto & impl : *impls) {
            const std::string n = impl.name;
            if (impl.provider == GGML_TRITON_PROVIDER_TRITON && n.find("fp16") != std::string::npos) {
                found_triton_rms_norm_fp16 = true;
                std::printf("found RMS_NORM impl: name=%s provider=%d priority=%d\n",
                            impl.name, (int) impl.provider, impl.priority);
            }
            if (impl.provider == GGML_TRITON_PROVIDER_TRITON && n.find("fp32") != std::string::npos) {
                found_triton_rms_norm_fp32 = true;
                std::printf("found RMS_NORM impl: name=%s provider=%d priority=%d\n",
                            impl.name, (int) impl.provider, impl.priority);
            }
        }
    }
    if (!found_triton_rms_norm_fp16) {
        std::fprintf(stderr, "FAIL: triton AOT RMS_NORM fp16 provider not registered in global registry\n");
        return 4;
    }
    if (!found_triton_rms_norm_fp32) {
        std::fprintf(stderr, "FAIL: triton AOT RMS_NORM fp32 provider not registered in global registry\n");
        return 5;
    }
    std::printf("Assert 4 passed: triton AOT RMS_NORM fp16 + fp32 providers are registered\n");

    // Assert 5 (B.2): the Triton AOT RoPE provider (3 modes x 2 dtypes = 6
    // impls) must be registered for GGML_OP_ROPE. Mirrors B.1's Assert 4
    // pattern. The CPU fp32 RoPE provider already exists
    // (see ggml-triton-provider-cpu.cpp:793-800 — only NORMAL mode) but we are
    // specifically asserting that the *triton AOT* entries (NORMAL, NEOX,
    // MROPE x fp16/fp32) get added by the new
    // ggml-triton-provider-rope.{h,cpp} files (Task 8).
    constexpr const char * expected_rope[] = {
        "triton_rope_normal_fp16_sm80",
        "triton_rope_normal_fp32_sm80",
        "triton_rope_neox_fp16_sm80",
        "triton_rope_neox_fp32_sm80",
        "triton_rope_mrope_fp16_sm80",
        "triton_rope_mrope_fp32_sm80",
    };
    bool found_rope[6] = {false, false, false, false, false, false};
    if (auto * impls = reg.get_impls(GGML_OP_ROPE)) {
        for (const auto & impl : *impls) {
            if (impl.provider != GGML_TRITON_PROVIDER_TRITON) continue;
            for (int i = 0; i < 6; ++i) {
                if (std::string(impl.name).find(expected_rope[i]) != std::string::npos) {
                    found_rope[i] = true;
                }
            }
        }
    }
    for (int i = 0; i < 6; ++i) {
        if (!found_rope[i]) {
            std::fprintf(stderr, "FAIL: triton AOT RoPE impl %s not registered in global registry\n", expected_rope[i]);
            return 6;
        }
    }
    std::printf("Assert 5 passed: 6 triton AOT RoPE impls (NORMAL+NEOX+MROPE x fp16/fp32) registered\n");

    // Assert 6 (B.3): the Triton AOT FlashAttn provider (3 head_dim × 2 dtype
    // × 2 kernel = 12 impls) must be registered for GGML_OP_FLASH_ATTN_EXT.
    // Mirrors B.1's Assert 4 and B.2's Assert 5 patterns. The CPU FlashAttn
    // provider already exists (see ggml-triton-provider-cpu.cpp; covers cases
    // our supports() rejects) but we are specifically asserting that the
    // *triton AOT* entries (prefill/decode × hd{64,96,128} × fp16/fp32) get
    // added by the new ggml-triton-provider-flash-attn.{h,cpp} files.
    constexpr const char * expected_flash_attn[] = {
        "triton_flash_attn_prefill_hd64_fp16_sm80",
        "triton_flash_attn_prefill_hd64_fp32_sm80",
        "triton_flash_attn_prefill_hd96_fp16_sm80",
        "triton_flash_attn_prefill_hd96_fp32_sm80",
        "triton_flash_attn_prefill_hd128_fp16_sm80",
        "triton_flash_attn_prefill_hd128_fp32_sm80",
        "triton_flash_attn_decode_hd64_fp16_sm80",
        "triton_flash_attn_decode_hd64_fp32_sm80",
        "triton_flash_attn_decode_hd96_fp16_sm80",
        "triton_flash_attn_decode_hd96_fp32_sm80",
        "triton_flash_attn_decode_hd128_fp16_sm80",
        "triton_flash_attn_decode_hd128_fp32_sm80",
    };
    bool found_flash_attn[12] = {false, false, false, false, false, false, false, false, false, false, false, false};
    if (auto * impls = reg.get_impls(GGML_OP_FLASH_ATTN_EXT)) {
        for (const auto & impl : *impls) {
            if (impl.provider != GGML_TRITON_PROVIDER_TRITON) continue;
            for (int i = 0; i < 12; ++i) {
                if (std::string(impl.name).find(expected_flash_attn[i]) != std::string::npos) {
                    found_flash_attn[i] = true;
                }
            }
        }
    }
    for (int i = 0; i < 12; ++i) {
        if (!found_flash_attn[i]) {
            std::fprintf(stderr, "FAIL: triton AOT FlashAttn impl %s not registered in global registry\n", expected_flash_attn[i]);
            return 7;
        }
    }
    std::printf("Assert 6 passed: 12 triton AOT FlashAttn impls (prefill+decode × hd{64,96,128} × fp16/fp32) registered\n");

    std::printf("OK: registry test passed\n");
    return 0;
}
