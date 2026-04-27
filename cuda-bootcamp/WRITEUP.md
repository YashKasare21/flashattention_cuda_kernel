# Building FlashAttention from Scratch: What I Learned About GPU Memory

## The Problem I Set Out to Solve

Standard attention has a memory problem that has nothing to do with compute. The naive implementation materializes the full N×N attention score matrix in GPU global memory (HBM). At N=4096 with float32, that's 4096 × 4096 × 4 bytes = **67 MB written and read back every single forward pass** — and that's before you account for the softmax intermediate, the dropout mask, and the backward pass. The GPU spends most of its time waiting for memory transfers, not doing math.

This is the memory wall. The GPU has ~312 TFLOPS of compute on a T4 but only ~300 GB/s of HBM bandwidth. A kernel that writes 67 MB per call is bandwidth-bound by construction, regardless of how fast the arithmetic units are.

FlashAttention's insight is that you never need the full N×N matrix in memory at once. You can tile Q, K, and V into blocks, keep the working set in on-chip SRAM (which is ~100× faster than HBM), and compute the exact same output using an online softmax recurrence that maintains running statistics across tiles. Memory complexity drops from O(N²) to O(N). I wanted to understand this from the hardware up, so I implemented it in CUDA.

## What I Built

Three progressively optimized CUDA kernels for the FlashAttention forward pass, each targeting a specific hardware bottleneck identified by profiling. The kernels implement SRAM tiling with 32×64 tiles, the online softmax recurrence (maintaining running max `m_i` and normalizer `l_i` across KV tiles), two-level causal masking (tile-level skip + element-level -∞), and 4D multi-head support via a 3D CUDA grid. All three are compiled as PyTorch C++ extensions and validated against `F.scaled_dot_product_attention`.

## The Optimization Journey

### V1 → V2: Killing the Bank Conflicts

Shared memory on a GPU is divided into 32 banks. When 32 threads in a warp all access addresses that map to the same bank, the hardware serializes those accesses — 32 sequential reads instead of 1 parallel one. This is a bank conflict, and it's invisible unless you profile.

V1 declared `float s_Q[32][64]`. With 64 floats per row and 32 banks, every row starts at a multiple of 32 floats, so `s_Q[0][k]`, `s_Q[1][k]`, ..., `s_Q[31][k]` all map to bank `k % 32`. When the compute loop accesses column `k` across all 32 threads simultaneously, all 32 threads hit the same bank. Nsight Compute confirmed this: **16,896,946 shared memory store conflicts** per kernel launch.

The fix is one integer: change the declaration to `float s_Q[32][65]`. Adding one float of padding shifts each row by 4 bytes, so row `i` starts at offset `i × 65 × 4` bytes. Now `s_Q[i][k]` maps to bank `(i*65 + k) % 32`, which spreads accesses across all 32 banks. Store conflicts dropped to **~8,700** — a 99.95% reduction.

The second V2 change replaced `reinterpret_cast<float4*>` loads (which require 16-byte pointer alignment and were causing `cudaErrorMisalignedAddress` on non-aligned slice pointers) with `__ldg()` scalar loads. `__ldg` routes through the read-only texture cache, bypasses L1, and has no alignment requirement beyond the natural 4-byte alignment of float.

Result: **2.976ms → 2.038ms, 1.46× faster**.

### V2 → V3: Engaging Tensor Cores

Tensor Cores are dedicated matrix multiply hardware on NVIDIA GPUs. A single `mma_sync` instruction executes a 16×16×16 FP16 matrix multiply-accumulate — that's 16×16×16×2 = 8,192 multiply-add operations in one warp-level instruction, compared to 256 scalar FMAs for the same computation. The theoretical throughput ratio is ~8×.

The `nvcuda::wmma` API exposes this through fragment types: `fragment<matrix_a>` and `fragment<matrix_b>` hold fp16 input tiles distributed across the 32 threads of a warp in a hardware-defined layout; `fragment<accumulator>` holds the fp32 result. You call `load_matrix_sync` to fill a fragment from shared memory, `mma_sync` to multiply, and `store_matrix_sync` to write the result back.

For V3, the 32×64 Q and K tiles are decomposed into a 2×2 grid of 16×16 output tiles, each accumulated over D=64 in 4 steps of 16. Q and K are cast from fp32 to fp16 on load; V and O stay fp32.

There was one tradeoff. `wmma::load_matrix_sync` requires 32-byte aligned shared memory pointers. The `+1` padding from V2 made each `__half` row 130 bytes wide — not divisible by 32 — breaking alignment at every 16-row sub-tile boundary. Removing the padding from `s_Q` and `s_K` restored alignment but partially reintroduced bank conflicts (~947K load, ~2.6M store). The Tensor Core throughput gain wins that tradeoff: V3 is 1.66× faster than V2 despite the conflict regression.

Result: **2.038ms → 1.231ms, 2.42× over V1**.

## What The Numbers Actually Mean

PyTorch SDPA runs in 0.401ms on the same hardware. V3 is 3× slower.

This gap is not a bug or an oversight — it's structural. SDPA uses `cp.async` instructions to prefetch the next KV tile into shared memory while the current tile's matrix multiply is executing, hiding memory latency behind compute. It also launches multiple warps per block, keeping the SM occupied while some warps stall on memory. V3 uses one warp per block by design (wmma requires all 32 threads in a warp to execute collectively; with BLOCK_SIZE=32 there is exactly one warp). Closing this gap would require redesigning the kernel around multi-warp execution and async prefetch pipelines — a substantially more complex undertaking.

The point of this project was not to beat PyTorch. It was to understand why PyTorch is fast. After profiling three kernel versions with Nsight Compute, I have a concrete answer: SDPA is fast because it hides memory latency through software pipelining and keeps the hardware fully occupied. Those are engineering decisions, not magic.

## What I Learned

**1. GPU performance is about data movement, not compute.** Every bottleneck I hit — the O(N²) HBM writes in naive attention, the bank conflicts in V1, the alignment constraints in V3 — was a memory problem. The arithmetic units were idle most of the time. The memory wall is real and it shows up in the profiler as ~8.5% SM throughput on a kernel that should be doing significant work.

**2. Nsight Compute is the only way to know what's actually happening.** Before profiling V1, I assumed the bottleneck was the scalar dot product loop. It wasn't — it was 16.8 million serialized shared memory accesses that I had no intuition about. Every optimization decision in this project came from a specific Nsight metric, not from reasoning about the code.

**3. Hardware constraints compose.** Fixing bank conflicts (V2) required padding that broke wmma alignment (V3). Fixing wmma alignment reintroduced bank conflicts. There is no globally optimal solution — only a sequence of informed tradeoffs where you measure the cost of each constraint and pick the one that loses less. The `+1` padding was the right call for V2; removing it was the right call for V3. The profiler tells you which tradeoff wins.

## Tools and Stack

- **CUDA C++** for all three kernel implementations, compiled with nvcc targeting sm_75.
- **PyTorch C++ extensions** (`torch.utils.cpp_extension`) to load the kernels as Python-callable modules without writing a full binding layer.
- **NVIDIA Nsight Compute** (`ncu`) for hardware-level profiling — SM throughput, memory bandwidth, shared memory bank conflicts, and sector utilization.
- **Google Colab T4** (16GB, compute capability 7.5) as the development and benchmarking environment.
- **Claude** (Kiro CLI) for implementation assistance and iteration on the kernel code.
