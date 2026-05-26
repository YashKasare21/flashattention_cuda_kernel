# FlashAttention CUDA Kernel — From Scratch

**Hardware-aware FlashAttention forward + backward pass implemented in pure CUDA C++**

[![CUDA](https://img.shields.io/badge/CUDA-12%2B-76B900?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-zone)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

> Built a hardware-aware FlashAttention CUDA kernel from scratch — SRAM tiling, online softmax, Tensor Cores, multi-warp blocks, and exact backward pass gradients. V4 forward approaches PyTorch SDPA speed on Tesla T4. V5 backward delivers O(N) memory with exact numerical match.

---

## 🚀 Key Results

### Forward Pass (Tesla T4, B=2, H=4, d=64, causal)

| Version | Seq=256 | Seq=512 | Seq=1024 | Seq=2048 | Seq=4096 | vs SDPA | Memory |
|---------|---------|---------|----------|----------|----------|---------|--------|
| V1 Baseline | 1.02 ms | 2.92 ms | 8.47 ms | 17.20 ms | 52.47 ms | 4.5× slower | O(N²) |
| V2 `__ldg` + padding | 0.30 ms | 0.89 ms | 3.05 ms | 11.47 ms | 44.32 ms | 3.8× slower | O(N²) |
| V3 Tensor Cores | 0.23 ms | 0.66 ms | 2.25 ms | 7.96 ms | 30.19 ms | 2.6× slower | O(N) |
| **V4 Multi-Warp** | **0.27 ms** | **0.66 ms** | **1.91 ms** | **6.37 ms** | **23.47 ms** | **2.0× slower** 🎯 | **O(N)** |
| PyTorch SDPA | 0.11 ms | 0.25 ms | 0.71 ms | 2.83 ms | 11.77 ms | 1.0× | — |

**V4 achieves 2× of SDPA speed with O(N) memory — the memory complexity reduction is the core win.**

### Backward Pass (Tesla T4, B=2, H=4, d=64, causal)

| Version | Seq=256 | Seq=512 | Seq=1024 | Seq=2048 | Seq=4096 | Memory |
|---------|---------|---------|----------|----------|----------|--------|
| **V5 Custom** | **9.39 ms** | **25.08 ms** | **87.47 ms** | **311.53 ms** | **1189.47 ms** | **O(N)** ✅ |
| PyTorch SDPA | 0.72 ms | 1.41 ms | 4.42 ms | 13.48 ms | 49.82 ms | O(N²) |

**V5 delivers exact dQ/dK/dV gradients (max diff < 0.002) with 2× memory reduction vs standard attention.**

---

## 📊 Benchmarks

### Forward Pass: V1 → V4 vs PyTorch SDPA
![Forward Benchmark](assets/benchmark_forward_v4.png)

### Backward Pass: V5 vs PyTorch SDPA
![Backward Benchmark](assets/benchmark_backward_v5.png)

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
| SMEM bank conflicts | 32-way | **~0** | minor | minor* |

*V3/V4 reintroduce minor conflicts to satisfy WMMA 32-byte alignment.

**V4 multi-warp design** (BLOCK_SIZE=64, 4 warps × 32 threads):
- Each warp owns 16 rows of the 64-row Q-tile
- All 128 threads cooperatively load K, V tiles into SMEM
- Each warp independently computes its 16-row strip of S via WMMA (4 × 16×16 fragments)
- Each warp runs online softmax + PV accumulation independently

SMEM budget (T4, 48 KB/SM): `s_Q(8KB) + s_K(8KB) + s_V(16KB) + s_S(16KB) = 48 KB` ✓

### Backward Pass (V5)

Recomputes attention weights P on-the-fly from saved `(M, L)` — no O(N²) storage.

```
S_ij  = dot(Q_i, K_j) * scale          # recomputed
P_ij  = exp(S_ij − M_i) / L_i          # recomputed from saved stats
D_i   = sum_d(dO_i · O_i)              # softmax backward correction
dV_j += P_ij * dO_i                    # atomicAdd
dS_ij = P_ij * (dot(dO_i, V_j) − D_i) * scale
dQ_i += dS_ij * K_j                    # local accumulation, no atomics
dK_j += dS_ij * Q_i                    # atomicAdd
```

**Memory design:** Q and dO rows held in registers (128 floats/thread). Only K and V tiles in SMEM (32 KB total).

---

## 🔧 Installation

```bash
git clone https://github.com/YashKasare21/flashattention_cuda_kernel.git
cd flashattention_cuda_kernel
pip install -e .
```

**Requirements:** CUDA Toolkit 12.0+, PyTorch 2.0+, Python 3.8+, SM75+ GPU (Tesla T4 or newer)

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
│   └── test_backward_v5.py       # V5 gradient correctness
├── benchmarks/
│   ├── benchmark.py              # V1–V4 head-to-head timing
│   └── benchmark_backward.py     # Forward+backward vs SDPA
├── assets/
│   ├── benchmark_forward_v4.png  # Forward scaling benchmark
│   ├── benchmark_backward_v5.png # Backward benchmark
│   └── architecture_diagram.png
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
