#pragma once

#include "ggml-triton-provider.h"

// Register all RMSNorm kernel providers into the given registry.
// Called during backend initialization (B.1 of docs/development/ROADMAP.md).
// Ships four impls: unweighted/weighted x fp16/fp32.
void ggml_triton_register_rmsnorm_providers(ggml_triton_op_registry & registry);
