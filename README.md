# FlashAttention-CUDA: From Scratch

A custom, hardware-aware implementation of the **FlashAttention forward pass** written in pure CUDA C++. The goal is to bypass the GPU **Memory Wall** — the bottleneck where standard attention forces the GPU to read and write an N×N matrix to slow global (HBM) memory for every forward pass.

---

## The Problem: Standard Attention is Memory-Bound

In standard attention, three separate kernels materialize the full N×N score matrix in global memory:

```
S = Q @ K^T / sqrt(d)   →  written to HBM  (N×N floats)
P = softmax(S)          →  read + written to HBM
O = P @ V               →  read from HBM
```

For N=1024 and fp32, that's ~4 MB of intermediate state per head per layer — read and written repeatedly. Memory bandwidth, not FLOPs, becomes the bottleneck.

---

## Architecture Highlights

### 1. SRAM Tiling & Online Softmax

The kernel **never materializes the N×N matrix**. Instead:

- Q, K, and V are loaded in `BLOCK_SIZE × d` tiles into **shared memory (SRAM)**, which is ~100× faster than HBM.
- The softmax is computed **online** (one tile at a time) using the FlashAttention recurrence:

```
m_new  = max(m_i,  max(S_tile))
l_new  = exp(m_i − m_new) · l_i  +  Σ exp(S_tile − m_new)
O_i   *= exp(m_i − m_new)                          # rescale old accumulator
O_i   += Σ_j exp(S_tile[j] − m_new) · V_tile[j]   # add new contribution
```

A single `O_i / l_i` division at the **end** normalizes the output — no per-tile division needed.

### 2. 4D Tensor Support — Batch & Multi-Head

Tensors are shaped `[B, H, N, d]`. The CUDA grid has **three dimensions**:

```
dim3 grid(N / BLOCK_SIZE,  H,  B)
         └── seq tile     └── head  └── batch

slice_offset = (blockIdx.z * H + blockIdx.y) * N * d
```

Each thread block resolves its own `(batch, head)` slice independently — zero inter-block communication.

### 3. Two-Level Causal Masking

Causal masking is applied at two granularities to minimize wasted work:

| Level | Condition | Action |
|---|---|---|
| **Tile-level** | `kv_tile > q_tile_idx` | `break` — entire future tile skipped, no shared-memory load |
| **Element-level** | `global_k_idx > global_q_idx` | `S_ij = -1e20f` — diagonal tile boundary handled per element |

The tile-level skip reduces the inner loop from O(N²) to **O(N²/2)** work. The `break` condition is uniform across all threads in a warp (`blockIdx.x` is shared), so there is **no warp divergence**.

---

## Repository Structure

```
flashattention_cuda_kernel/
├── cuda-bootcamp/
│   ├── src/
│   │   ├── flash_attn_forward.cu    # FlashAttention forward kernel (main)
│   │   ├── matmul.cu                # Naive matrix multiplication
│   │   ├── matmul_tiled.cu          # Tiled shared-memory matrix multiplication
│   │   └── vector_add.cu            # CUDA warm-up: vector addition
│   ├── tests/
│   │   ├── test_flash_attn.py       # Correctness + timing vs PyTorch SDPA
│   │   ├── test_matmul.py           # Naive vs tiled vs cuBLAS benchmark
│   │   └── online_softmax_test.py   # Online softmax math proof-of-concept
│   ├── benchmarks/
│   │   └── run_benchmarks.py        # Seq-len scaling benchmark + plot
│   ├── assets/
│   │   └── benchmark_seq_len.png    # Generated benchmark plot
│   └── setup.py
└── README.md
```

---

## Performance

Benchmark: `B=2, H=4, d=64`, sequence lengths 256 → 4096, causal attention, 30 iterations.

![Benchmark](cuda-bootcamp/assets/benchmark_seq_len.png)

---

## Usage

### Install

```bash
git clone https://github.com/YashKasare21/flashattention_cuda_kernel.git
cd flashattention_cuda_kernel/cuda-bootcamp
pip install -e . --no-build-isolation
```

### Run Tests

```bash
cd cuda-bootcamp
python tests/test_flash_attn.py
```

Expected output:
```
Shape: Q/K/V = [2, 4, 1024, 64]  |  iterations = 20
Max abs diff vs causal SDPA reference: <value>
Results match (atol=1e-3):             True

Custom causal FlashAttn:  X.XXX ms
PyTorch causal SDPA:      X.XXX ms
Speedup (Custom/SDPA):    X.XXx
```

### Run Benchmarks

```bash
cd cuda-bootcamp
python benchmarks/run_benchmarks.py
# Plot saved to cuda-bootcamp/assets/benchmark_seq_len.png
```

---

## Kernel Parameters

| Constant | Value | Notes |
|---|---|---|
| `BLOCK_SIZE` | 32 | Threads per block / tile width |
| `D_DIM` | 64 | Head dimension (compile-time constant) |
| Shared memory | ~24 KB/block | `s_Q + s_K + s_V = 3 × 32 × 64 × 4 B` |

---

## Background Reading

- [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135) — Dao et al., 2022
- [FlashAttention-2](https://arxiv.org/abs/2307.08691) — Dao, 2023
