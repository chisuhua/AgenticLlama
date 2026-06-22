// ggml/src/ggml-triton/ggml-triton-provider-flash-attn.h
//
// B.3 (FlashAttn provider) — see docs/development/ROADMAP.md §3 Phase B.3.
//
// Mirrors ggml-triton-provider-rope.h / ggml-triton-provider-rmsnorm.h:
// one free C++ registration function to be called from both
// ggml-triton-provider.cpp (global registry) and ggml-triton.cpp
// (per-context registry).  Ships 12 impls: prefill+decode ×
// head_dim ∈ {64, 96, 128} × fp16/fp32.

#pragma once

#include "ggml-triton-provider.h"

// Register all FlashAttn kernel providers into the given registry.
// Called during backend initialization (B.3 of docs/development/ROADMAP.md).
// Ships 12 impls: prefill+decode × hd{64,96,128} × fp{16,32}.
void ggml_triton_register_flash_attn_providers(ggml_triton_op_registry & registry);