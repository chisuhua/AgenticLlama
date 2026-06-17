// ggml/src/ggml-triton/ggml-triton-provider-rope.h
//
// B.2 (RoPE provider) — see docs/development/ROADMAP.md §3 Phase B.2.
//
// Mirrors ggml-triton-provider-cutlass.h / ggml-triton-provider-rmsnorm.h:
// one free C++ registration function to be called from both
// ggml-triton-provider.cpp (global registry) and ggml-triton.cpp
// (per-context registry).  Ships 6 impls: NORMAL+NEOX+MROPE × fp16/fp32.

#pragma once

#include "ggml-triton-provider.h"

// Register all RoPE kernel providers into the given registry.
// Called during backend initialization (B.2 of docs/development/ROADMAP.md).
// Ships 6 impls: NORMAL+NEOX+MROPE × fp16+fp32.
void ggml_triton_register_rope_providers(ggml_triton_op_registry & registry);
