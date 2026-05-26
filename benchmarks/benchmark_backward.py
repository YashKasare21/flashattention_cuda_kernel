"""
benchmark_backward.py — Forward+backward timing: V4+V5 vs PyTorch SDPA.

Measures:
  - Custom: V4 forward + V5 backward
  - PyTorch: SDPA forward + autograd backward
  - Memory: peak allocated (proxy for O(N) vs O(N²))
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn.functional as F
from functional import flash_attention

B, H, D = 2, 4, 64
SEQ_LENS = [256, 512, 1024, 2048, 4096]
WARMUP, ITERS = 10, 30


def bench(fn):
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(ITERS):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / ITERS


def peak_mem_mb(fn):
    torch.cuda.reset_peak_memory_stats()
    fn()
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / 1024**2


print(f"\nFlashAttention V5 Backward Benchmark  [B={B}, H={H}, d={D}]")
print(f"{'N':>6}  {'Custom fwd+bwd':>16}  {'SDPA fwd+bwd':>14}  {'Speedup':>9}  {'Custom mem':>12}  {'SDPA mem':>10}")
print("-" * 80)

for N in SEQ_LENS:
    torch.manual_seed(0)
    Q = torch.randn(B, H, N, D, device='cuda', requires_grad=True)
    K = torch.randn(B, H, N, D, device='cuda', requires_grad=True)
    V = torch.randn(B, H, N, D, device='cuda', requires_grad=True)
    dO = torch.randn(B, H, N, D, device='cuda')

    def custom_fwd_bwd():
        Q.grad = K.grad = V.grad = None
        O = flash_attention(Q, K, V)
        O.backward(dO)

    def sdpa_fwd_bwd():
        Q.grad = K.grad = V.grad = None
        O = F.scaled_dot_product_attention(Q, K, V, is_causal=True)
        O.backward(dO)

    t_custom = bench(custom_fwd_bwd)
    t_sdpa   = bench(sdpa_fwd_bwd)
    mem_c    = peak_mem_mb(custom_fwd_bwd)
    mem_s    = peak_mem_mb(sdpa_fwd_bwd)
    speedup  = t_sdpa / t_custom

    print(f"{N:>6}  {t_custom:>14.3f}ms  {t_sdpa:>12.3f}ms  {speedup:>8.2f}x  "
          f"{mem_c:>10.1f}MB  {mem_s:>8.1f}MB")
