# PROGRESS

## Pending

- `src/matmul.cu`, `src/matmul_tiled.cu`, `src/vector_add.cu` are orphaned — not
  registered in `setup.py`, and their former tests (`test_matmul.py`, root
  `test.py`) were removed. Either wire them back into `setup.py` + refresh tests,
  or delete them.
- Archived benchmark/profiling docs live in `docs/archive/` (recovered from
  `origin/feature/backward-pass`). Numbers there reflect the V1–V3 era; V4/V5
  numbers are only in `README.md` and `assets/` PNGs.
- Raw benchmark/timing results (`.csv`/`.json`) are not persisted anywhere —
  the README tables are not reproducible from saved data. Paper requires saving
  run-by-run data + multi-seed averaging before citing.
- No non-causal path and no D=128 benchmark coverage (V3 supports D=128 but it
  is untested/unbenchmarked).