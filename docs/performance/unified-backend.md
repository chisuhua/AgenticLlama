# Unified Triton Backend: Performance Notes

This document records the perf measurements taken when the standalone
`ggml-cutlass` and `ggml-tilelang` backends were folded into
`ggml-triton` as kernel providers.

## Test environment

| Item | Value |
| --- | --- |
| Host (this run) | no-GPU dev box (CUDA toolchain available, no device) |
| CPU-only Triton build | `build-cpu-triton` — compiled, ctest 91% pass (41/45, 4 pre-existing env failures) |
| GPU runs (this plan) | **deferred — requires a host with at least one NVIDIA GPU** |

## Status

The full perplexity / `llama-bench` matrix from the original Phase 5.1 spec
requires four GPU build modes and a model file. None of those are
available in the no-GPU environment this plan was executed in. The
results table below is **theoretical** — derived from the refactor's
mechanics, not from measurements. It must be filled in on a GPU host
before this plan can be considered fully verified.

| Backend | Perplexity | pp512 t/s | tg128 t/s |
| --- | --- | --- | --- |
| CUTLASS (standalone, deprecated) | _deferred to GPU host_ | _deferred_ | _deferred_ |
| CUTLASS via Triton provider | _deferred to GPU host_ | _deferred_ | _deferred_ |
| TileLang (standalone, deprecated) | _deferred to GPU host_ | _deferred_ | _deferred_ |
| TileLang via Triton provider | _deferred to GPU host_ | _deferred_ | _deferred_ |

See `.omo/plans/2026-06-08-unify-triton-multi-kernel-backend.md`
Phase 5.1 and `docs/superpowers/plans/baseline-deferral.md` for the
reproduction commands.

## What was actually measured (CPU-only)

- `build-cpu-triton` configured with `-DGGML_TRITON=ON
  -DGGML_TRITON_CPU_ONLY=ON`. Configure output included the expected
  `ggml-triton: CPU-only mode enabled` warning.
- Full build: 100% of targets compiled.
- `ctest -L main --timeout 900`: 41/45 tests passed.
  - 4 pre-existing env failures (network-dependent `test-download-model`,
    `test-llama-archs` SKIP rows, `test-thread-safety` blocked by
    `test-download-model`, `test-gguf`). None related to the Triton
    provider registry.
- The provider-registry smoke test (`tests/test-triton-registry.cpp`)
  was authored and staged; running it requires a configure that sets
  `-DGGML_TRITON_WITH_TILELANG=ON`, which is a GPU-capable configure.
  Defer to GPU host.

## Conclusion (theoretical, by construction)

The refactor is **correctness-preserving by construction**, on three
independent grounds:

1. **Kernel source files are byte-identical**. The Phase 4a step
   copied `ggml/src/ggml-cutlass/kernels/` and
   `ggml/src/ggml-tilelang/kernels/` into
   `ggml/src/ggml-triton/kernels/{cutlass,tilelang}/`. `diff -r` of the
   source vs. destination returned no output (byte-identical). The
   `.cu` files that the JIT/AOT compiler feeds are unchanged.

2. **Launcher C ABI is unchanged**. The TileLang provider's
   `ggml-triton-provider-tilelang.h` (Phase 1.1) declares
   `tilelang_add_fp16(void *, void *, void *, int32_t, cudaStream_t)`,
   which matches the existing `ggml-tilelang/kernels/include/tilelang_kernels.h`
   declaration (Phase 1.2 fix). The CUTLASS provider's launchers
   (`cutlass_kernels.h`) were already in the correct path before the
   refactor.

3. **Provider dispatch is at op-granularity, not device-granularity**.
   The new path selects a kernel within the same device context
   (`ggml_backend_triton`); the old path selected a different device
   (`ggml_backend_cutlass` or `ggml_backend_tilelang`). On the same
   GPU with the same kernel, the work performed is identical. The
   only theoretical overhead is the provider `select()` function
   call, which is `O(impls_for_op)` against a `std::vector` — well
   under microsecond at the op count in real graphs.

**Expected outcome on a GPU host**: perplexity identical (within float
noise), throughput within 3% of the standalone baseline. This is the
gate required before Phase 4 (deletion) is considered safe to merge.

## Reproduction

See `docs/superpowers/plans/baseline-deferral.md` for the full
reproduction recipe on a GPU host.
