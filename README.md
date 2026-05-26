# FlashAttention CUDA Kernel — From Scratch

**Hardware-aware FlashAttention forward + backward pass in pure CUDA C++, built iteratively from a scalar baseline to Tensor Core multi-warp production kernel.**

![CUDA](https://img.shields.io/badge/CUDA-12%2B-76B900?logo=nvidia&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-blue.svg)

---

## 🚀 Key Results

**Hardware:** Tesla T4 (SM75) · **Config:** B=2, H=4, N=1024, d=64, causal

| Version | Time (ms) | Speedup vs V1 | Key Technique | Memory |
|---------|-----------|---------------|---------------|--------|
| V1 Baseline | 2.976 | 1.00× | Scalar FMAs | O(N) |
| V2 `__ldg` + padding | 2.038 | 1.46× | Texture cache + bank-conflict fix | O(N) |
| V3 Tensor Cores | 1.231 | 2.42× | WMMA fp16 QKᵀ | O(N) |
| **V4 Multi-Warp** | **~1.0** | **~3×** | 4 warps × 128 threads | **O(N)** |
| PyTorch SDPA | 0.401 | 7.43× | Optimized reference | — |
| **V5 Backward** | — | — | Recompute P, exact gradients | **O(N)** |

V4 forward matches PyTorch SDPA within numerical tolerance. V5 backward delivers exact dQ/dK/dV with O(N) memory — no O(N²) attention matrix stored.

---

## 🏗️ Architecture

### Forward Pass (V1 → V4)

The core insight: standard attention is **memory-bound**, not compute-bound. The naive O(N²) attention matrix forces repeated HBM reads/writes. FlashAttention eliminates this by tiling Q, K, V into SRAM and fusing softmax with the matmul via an **online recurrence**.

**Online softmax recurrence** (maintained per Q-tile, across KV tiles):
```
m_new = max(m_i, max(S_tile))
l_new = exp(m_i − m_new) · l_i + Σ exp(S_tile − m_new)
O_i  *= exp(m_i − m_new)
O_i  += Σ_j exp(S_tile[j] − m_new) · V_tile[j]
```
Final output: `O_i / l_i`. Exact — no approximation.

**Optimization progression:**

| | V1 | V2 | V3 | V4 |
|---|---|---|---|---|
| BLOCK_SIZE | 32 | 32 | 32 | **64** |
| Threads/block | 32 | 32 | 32 | **128** |
| QKᵀ compute | scalar FMA | scalar FMA | **WMMA fp16** | WMMA fp16 |
| Global loads | plain | **`__ldg`** | `__ldg` | `__ldg` |
| SMEM bank conflicts | 32-way | **~0** | ~947K | ~2.6M* |
| SM Throughput (SOL) | ~8.5% | ~12% | ~8.5%† | — |

*V3/V4 reintroduce minor conflicts to satisfy WMMA 32-byte alignment.  
†Nsight `sm__throughput` tracks scalar cores only; Tensor Core execution is on a separate unit.

**V4 multi-warp design** (BLOCK_SIZE=64, 4 warps):
- Each warp owns 16 rows of the 64-row Q-tile
- Phase 1: all 128 threads cooperatively load Q into SMEM
- Phase 2: all 128 threads cooperatively load K, V into SMEM  
- Phase 3: each warp computes its 16-row strip of S via WMMA (1×4 tile of 16×16 fragments)
- Phase 4: each warp runs online softmax + PV accumulation independently

SMEM budget (T4, 48 KB/SM): `s_Q(8KB) + s_K(8KB) + s_V(16KB) + s_S(16KB) = 48 KB` ✓

**Two-level causal masking:**
- Tile-level: `kv_tile > q_tile_idx → break` — skips ~51% of tiles for N=1024
- Element-level: `k_idx > q_idx → S_ij = -∞` — handles diagonal tile boundaries

### Backward Pass (V5)

Recomputes attention weights P on-the-fly from saved `(M, L)` — no O(N²) storage.

**Algorithm** (per Q-tile, iterating over causal KV tiles):
```
S_ij  = dot(Q_i, K_j) * scale          # recomputed
P_ij  = exp(S_ij − M_i) / L_i          # recomputed from saved stats
D_i   = sum_d(dO_i · O_i)              # softmax backward correction
dV_j += P_ij * dO_i                    # atomicAdd
dP_ij = dot(dO_i, V_j)
dS_ij = P_ij * (dP_ij − D_i) * scale  # scale propagated back through S=QK^T*scale
dQ_i += dS_ij * K_j                    # local accumulation, no atomics
dK_j += dS_ij * Q_i                    # atomicAdd
```

**Memory design:** Q and dO rows held in registers (128 floats/thread). Only K and V tiles in SMEM (32 KB). Total SMEM: 32 KB — well within 48 KB limit.

---

## 📊 Benchmarks

![Scaling Benchmark](assets/benchmark_seq_len.png)

Sequence-length scaling (B=2, H=4, d=64, causal, T4 GPU). Custom kernel maintains constant performance ratio vs PyTorch SDPA across all lengths, confirming O(N) memory scaling.

---

## 🔧 Installation

```bash
git clone https://github.com/YashKasare21/flashattention_cuda_kernel.git
cd flashattention_cuda_kernel
pip install -e .
```

**Requirements:** CUDA Toolkit 12.0+, PyTorch 2.0+, Python 3.8+, Tesla T4 or SM75+ GPU

---

## 🧪 Testing

```bash
# Forward correctness (V1–V4 vs PyTorch SDPA)
python tests/test_flash_attn.py
python tests/test_flash_attn_v4.py

# Backward correctness (V5 dQ/dK/dV vs SDPA backward)
python tests/test_backward_v5.py
```

Expected backward output:
```
[test_backward_kernel] B=2 H=4 N=512 D=64
  dQ: max_diff=X.XXe-03  thr=1e-02  ✓
  dK: max_diff=X.XXe-03  thr=5e-02  ✓
  dV: max_diff=X.XXe-04  thr=5e-02  ✓
```

### Autograd usage

```python
from functional import flash_attention
import torch

Q = torch.randn(2, 8, 1024, 64, device='cuda', requires_grad=True)
K = torch.randn(2, 8, 1024, 64, device='cuda', requires_grad=True)
V = torch.randn(2, 8, 1024, 64, device='cuda', requires_grad=True)

O = flash_attention(Q, K, V)   # causal, differentiable
O.sum().backward()             # dQ, dK, dV populated
```

---

## 📁 Project Structure

```
flashattention_cuda_kernel/
├── src/
│   ├── flash_attn_v1.cu          # Baseline scalar FMA
│   ├── flash_attn_v2.cu          # __ldg + SMEM padding
│   ├── flash_attn_v3.cu          # WMMA Tensor Cores (1 warp)
│   ├── flash_attn_v4.cu          # Multi-warp production forward
│   ├── flash_attn_backward_v5.cu # Backward pass (O(N) memory)
│   ├── matmul.cu                 # Naive matmul baseline
│   ├── matmul_tiled.cu           # Tiled shared-memory matmul
│   └── vector_add.cu             # CUDA warm-up kernel
├── tests/
│   ├── test_flash_attn.py        # V1 correctness vs SDPA
│   ├── test_flash_attn_v2.py     # V2 correctness
│   ├── test_flash_attn_v3.py     # V3 correctness
│   ├── test_flash_attn_v4.py     # V4 correctness + benchmark
│   ├── test_backward_v5.py       # V5 gradient correctness
│   ├── test_matmul.py            # Matmul vs cuBLAS
│   └── online_softmax_test.py    # Online softmax math verification
├── benchmarks/
│   ├── benchmark.py              # V1–V4 head-to-head timing
│   └── benchmark_backward.py     # Forward+backward vs SDPA
├── assets/
│   ├── benchmark_seq_len.png     # Scaling benchmark plot
│   └── architecture_diagram.png  # System architecture
├── functional.py                 # PyTorch autograd wrapper
├── setup.py                      # Build all 5 CUDA extensions
├── requirements.txt
└── LICENSE
```

---

## 🎯 What I Learned

- **GPU memory hierarchy:** HBM → L2 → SMEM → registers, and how to exploit each tier
- **Memory-bound optimization:** coalesced access patterns, SMEM bank conflict elimination via `+1` column padding
- **Tensor Core programming:** WMMA fragment API, 32-byte alignment requirements, accumulator management
- **CUDA profiling:** Nsight Compute SOL analysis, identifying bottlenecks (uncoalesced loads, 32-way bank conflicts)
- **Algorithmic optimization:** online softmax recurrence, tiled backward pass with on-the-fly P recomputation

---

## 📚 References

- Dao et al. (2022). *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness.* NeurIPS 2022. [arXiv:2205.14135](https://arxiv.org/abs/2205.14135)
- Dao (2023). *FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning.* ICLR 2024. [arXiv:2307.08691](https://arxiv.org/abs/2307.08691)
- NVIDIA. *CUDA C++ Programming Guide.* [docs.nvidia.com](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

**Built by [Yash Kasare](https://github.com/YashKasare21) · Mumbai, India**
