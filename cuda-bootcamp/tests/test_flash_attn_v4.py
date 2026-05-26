"""
test_flash_attn_v4.py — Correctness + benchmark for V4 multi-warp FlashAttention.

V4 uses BLOCK_SIZE=64 (4 warps × 16 rows), so sequence lengths must be
divisible by 64. Tests use N=1024 (divisible by both 32 and 64).
"""

import torch
import torch.nn.functional as F
import custom_flash_attn_v3
import custom_flash_attn_v4

B, H, N, D = 2, 4, 1024, 64
WARMUP, ITERS = 10, 50

torch.manual_seed(42)
device = "cuda"

Q = torch.randn(B, H, N, D, device=device)
K = torch.randn(B, H, N, D, device=device)
V = torch.randn(B, H, N, D, device=device)

# ── Correctness ───────────────────────────────────────────────────────────────
ref    = F.scaled_dot_product_attention(Q, K, V, is_causal=True)
out_v4, _, _ = custom_flash_attn_v4.flash_attn_v4_forward(Q, K, V)

max_diff = (out_v4 - ref).abs().max().item()
match    = max_diff < 1e-2
print(f"\nV4 correctness vs causal SDPA:")
print(f"  Max abs diff : {max_diff:.6f}")
print(f"  Match (1e-2) : {match}")
assert match, f"V4 output mismatch: max diff {max_diff:.6f} > 1e-2"

# ── Benchmark helper ──────────────────────────────────────────────────────────
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

ms_v3   = bench(lambda: custom_flash_attn_v3.flash_attn_v3_forward(Q, K, V))
ms_v4   = bench(lambda: custom_flash_attn_v4.flash_attn_v4_forward(Q, K, V))
ms_sdpa = bench(lambda: F.scaled_dot_product_attention(Q, K, V, is_causal=True))

# ── Results table ─────────────────────────────────────────────────────────────
print(f"\n{'Method':<18} {'time_ms':>10} {'speedup_vs_v3':>15}")
print("-" * 46)
for name, ms in [
    ("V3 (wmma, 1w)",  ms_v3),
    ("V4 (wmma, 4w)",  ms_v4),
    ("Torch SDPA",     ms_sdpa),
]:
    speedup = ms_v3 / ms
    print(f"{name:<18} {ms:>10.3f} {speedup:>15.2f}x")

print(f"\nV4 speedup over V3: {ms_v3/ms_v4:.2f}x")
