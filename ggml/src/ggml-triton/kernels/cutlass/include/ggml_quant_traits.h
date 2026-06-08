#pragma once

#include <cstddef>
#include <cstdint>

// =============================================================================
// Mapping ggml quantisation block layouts to CUTLASS-friendly type traits.
//
// These traits describe the on-disk / in-memory layout of ggml quantised
// blocks (defined in ggml-common.h) so that downstream CUTLASS kernels and
// dequantisation prologues can reason about them generically.
//
// The struct definitions mirror the canonical block layouts from ggml:
//
//   block_q4_0 { ggml_half d;        uint8_t qs[QK4_0/2]; }   // 32 elements
//   block_q4_1 { ggml_half d, m;     uint8_t qs[QK4_1/2]; }   // 32 elements
//   block_q5_0 { ggml_half d;        uint8_t qh[4]; uint8_t qs[QK5_0/2]; }
//   block_q5_1 { ggml_half d, m;     uint8_t qh[4]; uint8_t qs[QK5_1/2]; }
//   block_q8_0 { ggml_half d;        int8_t  qs[QK8_0]; }      // 32 elements
//
// Each trait advertises the natural unit of work for a CUTLASS dequantisation
// prologue (kElementsPerBlock) and the storage footprint of the block in bytes
// (kBytesPerBlock). The prologue itself is implemented in a separate kernel
// translation unit; the traits intentionally stay header-only so they can be
// consumed from both .cu and .cpp files.
// =============================================================================

namespace ggml_cutlass {

// Element type categories visible to the CUTLASS kernels.
enum class ElementCategory {
    F32,
    F16,
    BF16,
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
    Unknown,
};

template <typename T>
struct ElementTraits;

template <> struct ElementTraits<float> {
    static constexpr ElementCategory kCategory      = ElementCategory::F32;
    static constexpr int             kBitsPerElement = 32;
    static constexpr bool            kIsQuantized    = false;
};

// We intentionally don't include <cuda_fp16.h> here so the header is usable in
// pure host C++ contexts. The forward-declared half is fine for trait queries.
struct half_storage { uint16_t bits; };
struct bfloat16_storage { uint16_t bits; };

template <> struct ElementTraits<half_storage> {
    static constexpr ElementCategory kCategory       = ElementCategory::F16;
    static constexpr int             kBitsPerElement = 16;
    static constexpr bool            kIsQuantized    = false;
};

template <> struct ElementTraits<bfloat16_storage> {
    static constexpr ElementCategory kCategory       = ElementCategory::BF16;
    static constexpr int             kBitsPerElement = 16;
    static constexpr bool            kIsQuantized    = false;
};

// -----------------------------------------------------------------------------
// Quantised block traits
// -----------------------------------------------------------------------------

struct GgmlQ4_0Traits {
    static constexpr ElementCategory kCategory          = ElementCategory::Q4_0;
    static constexpr int             kElementsPerBlock  = 32;
    // sizeof(ggml_half) + 16 packed nibbles = 2 + 16 = 18
    static constexpr int             kBytesPerBlock     = 18;
    static constexpr bool            kIsQuantized       = true;
    static constexpr bool            kHasMin            = false;
    static constexpr int             kBitsPerWeight     = 4;
};

struct GgmlQ4_1Traits {
    static constexpr ElementCategory kCategory          = ElementCategory::Q4_1;
    static constexpr int             kElementsPerBlock  = 32;
    // 2 * sizeof(ggml_half) + 16 packed nibbles = 4 + 16 = 20
    static constexpr int             kBytesPerBlock     = 20;
    static constexpr bool            kIsQuantized       = true;
    static constexpr bool            kHasMin            = true;
    static constexpr int             kBitsPerWeight     = 4;
};

struct GgmlQ5_0Traits {
    static constexpr ElementCategory kCategory          = ElementCategory::Q5_0;
    static constexpr int             kElementsPerBlock  = 32;
    // sizeof(ggml_half) + 4 (qh) + 16 (low nibbles) = 22
    static constexpr int             kBytesPerBlock     = 22;
    static constexpr bool            kIsQuantized       = true;
    static constexpr bool            kHasMin            = false;
    static constexpr int             kBitsPerWeight     = 5;
};

struct GgmlQ5_1Traits {
    static constexpr ElementCategory kCategory          = ElementCategory::Q5_1;
    static constexpr int             kElementsPerBlock  = 32;
    // 2 * sizeof(ggml_half) + 4 (qh) + 16 (low nibbles) = 24
    static constexpr int             kBytesPerBlock     = 24;
    static constexpr bool            kIsQuantized       = true;
    static constexpr bool            kHasMin            = true;
    static constexpr int             kBitsPerWeight     = 5;
};

struct GgmlQ8_0Traits {
    static constexpr ElementCategory kCategory          = ElementCategory::Q8_0;
    static constexpr int             kElementsPerBlock  = 32;
    // sizeof(ggml_half) + 32 int8 = 2 + 32 = 34
    static constexpr int             kBytesPerBlock     = 34;
    static constexpr bool            kIsQuantized       = true;
    static constexpr bool            kHasMin            = false;
    static constexpr int             kBitsPerWeight     = 8;
};

// Compile-time sanity checks against the published ggml block sizes. These
// numbers are stable parts of the ggml on-disk format.
static_assert(GgmlQ4_0Traits::kBytesPerBlock == 18, "block_q4_0 layout drift");
static_assert(GgmlQ4_1Traits::kBytesPerBlock == 20, "block_q4_1 layout drift");
static_assert(GgmlQ5_0Traits::kBytesPerBlock == 22, "block_q5_0 layout drift");
static_assert(GgmlQ5_1Traits::kBytesPerBlock == 24, "block_q5_1 layout drift");
static_assert(GgmlQ8_0Traits::kBytesPerBlock == 34, "block_q8_0 layout drift");

} // namespace ggml_cutlass
