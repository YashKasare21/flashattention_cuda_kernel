"""
test_flash_attn_v3.py — Correctness + benchmark for FlashAttention V1/V2/V3 vs PyTorch SDPA.

Correctness tolerance is relaxed to atol=1e-2 for V3 because fp16 QKᵀ compute
introduces small numerical differences vs the fp32 reference.
"""

import torch
import torch.nn.functional as F
import custom_flash_attn
import custom_flash_attn_v2
import custom_flash_attn_v3

B, H, N, D = 2, 4, 1024, 64
WARMUP, ITERS = 10, 50

torch.manual_seed(42)
device = "cuda"

Q = torch.randn(B, H, N, D, device=device)
K = torch.randn(B, H, N, D, device=device)
V = torch.randn(B, H, N, D, device=device)

# ── Correctness ───────────────────────────────────────────────────────────────
ref = F.scaled_dot_product_attention(Q, K, V, is_causal=True)
out_v3 = custom_flash_attn_v3.flash_attn_v3_forward(Q, K, V)

max_diff = (out_v3 - ref).abs().max().item()
match    = max_diff < 1e-2
print(f"\nV3 correctness vs causal SDPA:")
print(f"  Max abs diff : {max_diff:.6f}")
print(f"  Match (1e-2) : {match}")
assert match, f"V3 output mismatch: max diff {max_diff:.6f} > 1e-2"

# ── Benchmark helper ──────────────────────────────────────────────────────────
def bench(fn):
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)
    torch.cuda.reset_peak_memory_stats()
    start.record()
    for _ in range(ITERS):
        fn()
    end.record()
    torch.cuda.synchronize()
    ms  = start.elapsed_time(end) / ITERS
    mem = torch.cuda.max_memory_allocated() / 1024**2
    return ms, mem

ms_v1,   mem_v1   = bench(lambda: custom_flash_attn.flash_attn_forward(Q, K, V))
ms_v2,   mem_v2   = bench(lambda: custom_flash_attn_v2.flash_attn_v2_forward(Q, K, V))
ms_v3,   mem_v3   = bench(lambda: custom_flash_attn_v3.flash_attn_v3_forward(Q, K, V))
ms_sdpa, mem_sdpa = bench(lambda: F.scaled_dot_product_attention(Q, K, V, is_causal=True))

# ── Results table ─────────────────────────────────────────────────────────────
print(f"\n{'Method':<18} {'time_ms':>10} {'speedup_vs_v1':>15} {'peak_mem_MB':>13}")
print("-" * 60)
for name, ms, mem in [
    ("V1 (baseline)",  ms_v1,   mem_v1),
    ("V2 (__ldg+pad)", ms_v2,   mem_v2),
    ("V3 (wmma TC)",   ms_v3,   mem_v3),
    ("Torch SDPA",     ms_sdpa, mem_sdpa),
]:
    speedup = ms_v1 / ms
    print(f"{name:<18} {ms:>10.3f} {speedup:>15.2f}x {mem:>12.1f}")
