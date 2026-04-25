import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn.functional as F
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import custom_flash_attn

B, H, D = 2, 4, 64
SEQ_LENS = [256, 512, 1024, 2048, 4096]
ITERS    = 30
WARMUP   = 5


def bench(fn, iters, warmup):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000 / iters


flash_times = []
sdpa_times  = []

for N in SEQ_LENS:
    torch.manual_seed(0)
    Q = torch.randn(B, H, N, D, dtype=torch.float32, device='cuda')
    K = torch.randn(B, H, N, D, dtype=torch.float32, device='cuda')
    V = torch.randn(B, H, N, D, dtype=torch.float32, device='cuda')

    ft = bench(lambda: custom_flash_attn.flash_attn_forward(Q, K, V), ITERS, WARMUP)
    st = bench(lambda: F.scaled_dot_product_attention(Q, K, V, is_causal=True), ITERS, WARMUP)

    flash_times.append(ft)
    sdpa_times.append(st)
    print(f"N={N:5d}  |  Custom: {ft:7.3f} ms  |  SDPA: {st:7.3f} ms  |  Ratio: {ft/st:.2f}x")

# Plot
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(SEQ_LENS, flash_times, marker='o', linewidth=2, label='Custom FlashAttn (causal)')
ax.plot(SEQ_LENS, sdpa_times,  marker='s', linewidth=2, label='PyTorch SDPA (is_causal=True)')

ax.set_xlabel('Sequence Length', fontsize=13)
ax.set_ylabel('Execution Time (ms)', fontsize=13)
ax.set_title('FlashAttention-CUDA: Scaling vs PyTorch SDPA\n'
             f'[B={B}, H={H}, d={D}]', fontsize=14)
ax.set_xticks(SEQ_LENS)
ax.set_xticklabels([str(n) for n in SEQ_LENS])
ax.legend(fontsize=12)
ax.grid(True, linestyle='--', alpha=0.6)
fig.tight_layout()

out_path = os.path.join(os.path.dirname(__file__), '..', 'assets', 'benchmark_seq_len.png')
fig.savefig(out_path, dpi=150)
print(f"\nPlot saved → {os.path.abspath(out_path)}")
