"""
run_benchmarks_v4.py — V3 vs V4 scaling benchmark across sequence lengths.

V4 requires N divisible by 64 (BLOCK_SIZE=64).
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import custom_flash_attn_v3
import custom_flash_attn_v4

B, H, D = 2, 4, 64
# All must be divisible by 64 (V4 BLOCK_SIZE)
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


v3_times   = []
v4_times   = []
sdpa_times = []

print(f"{'N':>6}  {'V3 (ms)':>10}  {'V4 (ms)':>10}  {'SDPA (ms)':>10}  {'V4/V3':>8}")
print("-" * 55)

for N in SEQ_LENS:
    torch.manual_seed(0)
    Q = torch.randn(B, H, N, D, dtype=torch.float32, device='cuda')
    K = torch.randn(B, H, N, D, dtype=torch.float32, device='cuda')
    V = torch.randn(B, H, N, D, dtype=torch.float32, device='cuda')

    t3   = bench(lambda: custom_flash_attn_v3.flash_attn_v3_forward(Q, K, V))
    t4   = bench(lambda: custom_flash_attn_v4.flash_attn_v4_forward(Q, K, V))
    tsdp = bench(lambda: F.scaled_dot_product_attention(Q, K, V, is_causal=True))

    v3_times.append(t3)
    v4_times.append(t4)
    sdpa_times.append(tsdp)
    print(f"{N:>6}  {t3:>10.3f}  {t4:>10.3f}  {tsdp:>10.3f}  {t3/t4:>8.2f}x")

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(SEQ_LENS, v3_times,   marker='o', linewidth=2, label='V3 (wmma, 1 warp/block)')
ax.plot(SEQ_LENS, v4_times,   marker='s', linewidth=2, label='V4 (wmma, 4 warps/block)')
ax.plot(SEQ_LENS, sdpa_times, marker='^', linewidth=2, label='PyTorch SDPA (is_causal=True)')

ax.set_xlabel('Sequence Length', fontsize=13)
ax.set_ylabel('Execution Time (ms)', fontsize=13)
ax.set_title('FlashAttention V3 vs V4: Multi-Warp Scaling\n'
             f'[B={B}, H={H}, d={D}, T4 GPU]', fontsize=14)
ax.set_xticks(SEQ_LENS)
ax.set_xticklabels([str(n) for n in SEQ_LENS])
ax.legend(fontsize=12)
ax.grid(True, linestyle='--', alpha=0.6)
fig.tight_layout()

out_path = os.path.join(os.path.dirname(__file__), '..', 'assets', 'benchmark_v3_v4.png')
fig.savefig(out_path, dpi=150)
print(f"\nPlot saved → {os.path.abspath(out_path)}")
