# CUDA Implementation Notes

## Phase 3 — Tensor Cores (wmma API)

### What wmma fragments are

The `nvcuda::wmma` API exposes NVIDIA Tensor Core hardware through three fragment types:

- `fragment<matrix_a, 16,16,16, __half, row_major>` — holds a 16×16 fp16 A-matrix tile, distributed across the 32 threads of a warp in an opaque hardware layout.
- `fragment<matrix_b, 16,16,16, __half, col_major>` — holds a 16×16 fp16 B-matrix tile. `col_major` causes the load to transpose the tile in-place, so loading a row-major K sub-tile with `col_major` gives Kᵀ directly.
- `fragment<accumulator, 16,16,16, float>` — holds a 16×16 fp32 accumulator tile. Using fp32 here preserves numerical precision for the softmax and output accumulation steps.

`mma_sync(D, A, B, C)` computes `D = A × B + C` on Tensor Core hardware in a single warp-collective instruction.

### The 16×16×16 tile constraint

Each `mma_sync` call operates on exactly 16×16×16 tiles: A is 16×16, B is 16×16, the inner (K) dimension is 16. For larger matrices, tile the computation manually. In V3, `BLOCK_SIZE=32` and `D_DIM=64` are decomposed as:

- 2×2 output tiles of 16×16 cover the 32×32 score matrix S.
- Each output tile accumulates over `D_DIM=64` in 4 steps of 16 → 4 `mma_sync` calls per quadrant, 16 total per KV tile.

### Why fp16 input is required

Tensor Core MMA instructions on sm_75 (Turing / T4) require `__half` (fp16) operands for the A and B matrices. The accumulator can be fp32. In V3, Q and K tiles are cast from fp32 to fp16 on load into shared memory; V and O remain fp32 throughout, so the only precision loss is in the QKᵀ dot product, which introduces a max abs error of ~0.001 vs the fp32 reference (well within the atol=1e-2 tolerance).

### The 32-byte alignment requirement for load_matrix_sync

`wmma::load_matrix_sync` and `store_matrix_sync` require the shared memory pointer to be **32-byte (256-bit) aligned**. Two rules follow:

1. **No odd padding on `__half` arrays.** `D_DIM=64` halfs × 2 bytes = 128 bytes per row — already a multiple of 32. Adding `+1` padding makes it 130 bytes, breaking alignment at every 16-row sub-tile boundary (`&s_Q[16][0]` would be at offset 130×16 = 2080 bytes, not divisible by 32). Solution: zero padding for `__half` arrays.
2. **Explicit alignment on the accumulator array.** Declare `s_S` with `__shared__ __align__(32) float s_S[...]` to guarantee the base pointer is 32-byte aligned regardless of what precedes it in the shared memory layout.

The stride argument to `load_matrix_sync` must match the actual array row stride in memory (number of elements, not bytes). For unpadded `__half s_Q[32][64]` the stride is `64`; for `float s_S[32][32]` the stride is `32`.

### The 1-warp-per-block constraint at BLOCK_SIZE=32

`wmma` operations are **warp-collective**: all 32 threads in a warp must execute `load_matrix_sync`, `mma_sync`, and `store_matrix_sync` together (no divergence, no subset). With `BLOCK_SIZE=32`, each thread block contains exactly 32 threads = 1 warp, so this constraint is automatically satisfied. If `BLOCK_SIZE` were increased to 64 (2 warps), each warp would need its own independent fragment set and the tile decomposition would need to be redesigned accordingly.
