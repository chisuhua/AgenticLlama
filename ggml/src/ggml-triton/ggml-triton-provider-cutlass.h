#pragma once

#include "ggml-triton-provider.h"

// Register all CUTLASS kernel providers into the given registry.
// Called during backend initialization when GGML_TRITON_HAS_CUTLASS is defined.
void ggml_triton_register_cutlass_providers(ggml_triton_op_registry & registry);
