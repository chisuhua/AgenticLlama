# TileLang → Triton TileLang Provider Migration

`GGML_TILELANG` (the standalone TileLang backend) is deprecated. It will be
removed in the next release. Switch to the unified Triton backend with the
TileLang provider enabled.

## What changes

| Before | After |
| --- | --- |
| `-DGGML_TILELANG=ON` | `-DGGML_TRITON=ON -DGGML_TRITON_WITH_TILELANG=ON` |
| Backend name: `TileLang0` | Backend name: `Triton0` (same device, same kernels) |
| `ggml_backend_tilelang_reg()` | `ggml_backend_triton_reg()` |

## Why

Three backends (Triton, CUTLASS, TileLang) carry overlapping maintenance
for CUDA device management, buffer allocation, and dispatch. The
`ggml-triton` backend already implements a provider-pattern registry that
selects the best kernel for each op. The TileLang kernels are now a
provider inside Triton, so users get the same kernels with one backend
surface area. See `.omo/plans/2026-06-08-unify-triton-multi-kernel-backend.md`
for the full deprecation plan.

## Compatibility

- `GGML_TILELANG=ON` and `GGML_TRITON_WITH_TILELANG=ON` can both be
  enabled during the deprecation period. The standalone backend prints a
  CMake deprecation warning.
- Performance is unchanged: same TileLang AOT-compiled `.cu` files, same launchers.
- The change that *will* affect you: the device name moves from
  `TileLang0` to `Triton0`. Update any scripts that filter `ggml_backend_reg_name()`.

## Verifying

After switching, run:

```bash
./build/bin/llama-bench -m model.gguf -p 512 -n 128
```

and compare against your previous TileLang-standalone numbers. They
should be within 1%.
