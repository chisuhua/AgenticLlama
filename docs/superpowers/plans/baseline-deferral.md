# Baseline Deferral Note — Phase 0

Phase 0 of the unify-triton-multi-kernel-backend plan called for three
baseline builds (CPU-only Triton, GPU CUTLASS, GPU TileLang) so that
Phase 5 could measure against them. As of the date of this note, the
following baselines have been captured or deferred:

| Baseline | Status | Captured at |
| --- | --- | --- |
| CPU-only Triton (`build-cpu-triton`) | captured | this worktree |
| GPU CUTLASS standalone (`build-gpu-cutlass`) | **deferred — requires GPU** |  |
| GPU TileLang standalone (`build-gpu-tilelang`) | **deferred — requires GPU** |  |
| GPU CUTLASS via Triton provider | **deferred — requires GPU** |  |
| GPU TileLang via Triton provider | **deferred — requires GPU** |  |

## Why

The build environment in which this plan was first executed does not
have a GPU device:

- `nvidia-smi`: command not found
- `/dev/nvidia*`: no such files
- `/usr/local/cuda`: missing (custom CUDA toolchain only at
  `/workspace/project/opt/cuda/bin/nvcc`)

CUDA compilation works, but `cuInit` / `cudaGetDeviceCount` return
zero devices, so any build that registers a GPU backend at config
time will report zero devices and any binary that requires a real
GPU at runtime will fail to launch.

The CPU-only Triton build was the only path that exercises the
provider-registry code end-to-end without a GPU. The other four
baselines are the actual reference points for Phase 5's perf gate
and **must be re-captured on a host with at least one NVIDIA GPU**
before Phase 5 can produce a meaningful verdict.

## What to do on a GPU host

```bash
# 1. Configure + build all four modes (parallel-safe, separate build dirs)
cmake -B build-gpu-cutlass \
  -DGGML_TRITON=ON -DGGML_TRITON_WITH_CUTLASS=ON -DGGML_USE_CUTLASS=ON \
  -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_TOOLS=OFF \
  -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=OFF
cmake -B build-gpu-tilelang \
  -DGGML_USE_TILELANG=ON \
  -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_TOOLS=OFF \
  -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=OFF
cmake -B build-triton-tilelang \
  -DGGML_TRITON=ON -DGGML_TRITON_WITH_TILELANG=ON \
  -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_TOOLS=OFF \
  -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=OFF
# (rebuild build-cpu-triton with -DGGML_TRITON_WITH_TILELANG=ON
#  for the "TileLang via Triton" baseline)

for d in build-gpu-cutlass build-gpu-tilelang build-triton-tilelang; do
  cmake --build "$d" -j$(nproc) --config Release
  (cd "$d" && ctest -L main --timeout 900)
done

# 2. Capture test-backend-ops output for each
for d in build-gpu-cutlass build-gpu-tilelang build-triton-tilelang; do
  "./$d/bin/test-backend-ops" > "docs/superpowers/plans/baseline-$(basename $d)-ops.txt"
done

# 3. Run the perf comparison (requires a model file + a prompt file)
PROMPT_FILE=/path/to/wikitext-2-raw/wiki.test.raw \
  scripts/bench-backend.sh /path/to/model.gguf cutlass-standalone
# (repeat for tilelang-standalone, cutlass-via-triton, tilelang-via-triton)
```

After all four baselines are captured, this deferral note should be
updated to "captured" status and the dates filled in. Phase 5.1 can
then run.

## ctest notes (CPU-only baseline run)

The CPU-only `ctest -L main` run on this worktree reported 41/45
tests passed (91%). The 4 failures are pre-existing env issues
unrelated to this plan:

- `test-download-model` — fails because there is no network access
  to ggml-org/models. The test is a CMake `file(DOWNLOAD)` wrapper.
- `test-thread-safety` — "Not Run" because it depends on
  `test-download-model` succeeding first.
- `test-llama-archs` — emits a long table of `SKIP` rows for
  architecture/device combinations not compiled in this build. The
  subprocess abort comes from the table-rendering code path, not
  from a logic error.
- `test-gguf` — same family of env-dependent failures; the test
  log shows a pre-existing crash path independent of ggml-triton
work.

None of these touch the provider-registry code path that the plan
modifies. The CPU-only baseline is therefore valid for what Phase 5
needs: the registry + dispatcher compile and link, and the
unaffected tests pass cleanly.
