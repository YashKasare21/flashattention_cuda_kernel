# PROGRESS

## Pending

- `src/matmul.cu`, `src/matmul_tiled.cu`, `src/vector_add.cu` are orphaned — not
  registered in `setup.py`, and their former tests (`test_matmul.py`, root
  `test.py`) were removed. Either wire them back into `setup.py` + refresh tests,
  or delete them.
- `src/flash_attn_backward_v6.cu` is a WIP V6 backward kernel (WMMA Tensor Cores —
  addresses V5's scalar matmuls, global atomics, and 255-register spills).
  Committed as WIP (`5588400`), **GPU-untested**. The `store_matrix_sync` compile
  errors (fp32 accumulator → `__half*` destination) were fixed with an explicit
  `__float2half` conversion in `store_accum_half_smem`, and dQ is now flushed
  straight from fragments via `store_accum_global` (removed an out-of-bounds
  `s_Sp`→`float[64][64]` reinterpret). Still needs a build + correctness test
  (`tests/test_backward_v6.py`) + benchmark on a T4/Colab.
- Archived profiling/design docs live in `docs/archive/`
  (`WRITEUP.md`, `CUDA_NOTES.md`), recovered from
  `origin/feature/backward-pass`; their numbers reflect the V1–V3 era.
- No non-causal path anywhere (kernels hardcode causal masking). Forward D=128 is
  now benchmarkable (V3/SDPA), but the V3 D=128 code path still lacks a dedicated
  correctness test (only D=64 is exercised in `tests/`).

## Completed

- Benchmarked on Tesla T4 (15 repeats; `results/bench_20260805.json` committed) and
  refreshed the README "Key Results" tables from this verified data; the old
  hardcoded single-run numbers were demoted to a footnote as superseded.
  `docs/BACKWARD_ANALYSIS.md` recomputes its time× / throughput-gap table from the
  same dataset (`v4+v5 − v4` for backward-only time).
- Recovered WRITEUP/CUDA_NOTES/benchmark-script provenance into `docs/archive/`;
  merged the archived `run_benchmarks*.py` functionality into
  `benchmarks/benchmark.py` (forward scaling) and removed their duplicate
  archived copies.
- Rewrote `benchmarks/benchmark.py`, `benchmarks/benchmark_backward.py` to:
  mean ± std timing via CUDA events over `--repeats` rounds of `--iters`
  launches, TFLOPS/throughput, peak-memory (backward), auto-logged
  `device_name`, and structured JSON output.
- Added `benchmarks/run_all_benchmarks.py` (single entry point,
  `--output results/bench_YYYYMMDD.json`), Colab-friendly (no hardcoded paths;
  unbuilt kernels are skipped with a warning).
- Removed stale broken test files; fixed misleading README claims.