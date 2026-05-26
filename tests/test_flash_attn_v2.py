import torch
import torch.nn.functional as F
import custom_flash_attn
import custom_flash_attn_v2

B, H, N, D = 2, 4, 1024, 64
WARMUP, ITERS = 10, 50

torch.manual_seed(42)
device = torch.device("cuda")

Q = torch.randn(B, H, N, D, device=device)
K = torch.randn(B, H, N, D, device=device)
V = torch.randn(B, H, N, D, device=device)

# ── Correctness ───────────────────────────────────────────────────────────────
ref  = F.scaled_dot_product_attention(Q, K, V, is_causal=True)
out1 = custom_flash_attn.flash_attn_forward(Q, K, V)
out2 = custom_flash_attn_v2.flash_attn_v2_forward(Q, K, V)

assert torch.allclose(out1, ref, atol=1e-3), "V1 vs ref FAILED"
assert torch.allclose(out2, ref, atol=1e-3), "V2 vs ref FAILED"
print("Correctness: V1 PASS  V2 PASS")


def bench(fn, *args):
    """Return mean kernel time in ms using CUDA events."""
    for _ in range(WARMUP):
        fn(*args)
    torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats()
    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(ITERS):
        fn(*args)
    end.record()
    torch.cuda.synchronize()

    elapsed_ms = start.elapsed_time(end) / ITERS
    peak_mb    = torch.cuda.max_memory_allocated() / (1024 ** 2)
    return elapsed_ms, peak_mb


def sdpa(q, k, v):
    return F.scaled_dot_product_attention(q, k, v, is_causal=True)


t1, m1 = bench(custom_flash_attn.flash_attn_forward, Q, K, V)
t2, m2 = bench(custom_flash_attn_v2.flash_attn_v2_forward, Q, K, V)
t3, m3 = bench(sdpa, Q, K, V)

# ── Results table ─────────────────────────────────────────────────────────────
print()
print(f"{'Method':<20} {'time_ms':>10} {'speedup_vs_v1':>15} {'peak_mem_MB':>13}")
print("-" * 62)
for name, t, m in [("V1 (baseline)", t1, m1), ("V2 (float4+pad)", t2, m2), ("Torch SDPA", t3, m3)]:
    speedup = t1 / t
    print(f"{name:<20} {t:>10.3f} {speedup:>15.3f}x {m:>12.1f}")
