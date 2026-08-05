# PROGRESS

## Pending

- `src/matmul.cu`, `src/matmul_tiled.cu`, `src/vector_add.cu` are orphaned — not
  registered in `setup.py`, and their former tests (`test_matmul.py`, root
  `test.py`) were removed. Either wire them back into `setup.py` + refresh tests,
  or delete them.
- Archived profiling/design docs live in `docs/archive/`
  (`WRITEUP.md`, `CUDA_NOTES.md`), recovered from
  `origin/feature/backward-pass`; their numbers reflect the V1–V3 era.
- Benchmark scripts now persist JSON (run via
  `python benchmarks/run_all_benchmarks.py --output results/bench_YYYYMMDD.json`).
  **Next**: run on a T4/A100 GPU and commit the generated `results/` JSON so the
  README tables become reproducible (update README numbers from the new run).
- No non-causal path anywhere (kernels hardcode causal masking). Forward D=128 is
  now benchmarkable (V3/SDPA), but the V3 D=128 code path still lacks a dedicated
  correctness test (only D=64 is exercised in `tests/`).

## Completed

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