import torch
import torch.nn.functional as F
import time
import custom_flash_attn
try:
    import custom_flash_attn_v4
    HAS_V4 = True
except ImportError:
    HAS_V4 = False

B, H, N, D = 2, 4, 1024, 64
ITERS = 20

torch.manual_seed(42)
Q = torch.randn(B, H, N, D, dtype=torch.float32, device='cuda')
K = torch.randn(B, H, N, D, dtype=torch.float32, device='cuda')
V = torch.randn(B, H, N, D, dtype=torch.float32, device='cuda')

# Warmup
for _ in range(5):
    _ = custom_flash_attn.flash_attn_forward(Q, K, V)
    _ = F.scaled_dot_product_attention(Q, K, V, is_causal=True)
torch.cuda.synchronize()


def bench(fn, iters):
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000 / iters


flash_ms = bench(lambda: custom_flash_attn.flash_attn_forward(Q, K, V), ITERS)
sdpa_ms  = bench(lambda: F.scaled_dot_product_attention(Q, K, V, is_causal=True), ITERS)

# Correctness check
O_custom = custom_flash_attn.flash_attn_forward(Q, K, V)
O_ref    = F.scaled_dot_product_attention(Q, K, V, is_causal=True)
torch.cuda.synchronize()

match    = torch.allclose(O_custom, O_ref, atol=1e-3, rtol=1e-3)
max_diff = (O_custom - O_ref).abs().max().item()

print(f"Shape: Q/K/V = [{B}, {H}, {N}, {D}]  |  iterations = {ITERS}")
print(f"Max abs diff vs causal SDPA reference: {max_diff:.2e}")
print(f"Results match (atol=1e-3):             {match}")
print()
print(f"Custom causal FlashAttn:  {flash_ms:.3f} ms")
print(f"PyTorch causal SDPA:      {sdpa_ms:.3f} ms")
print(f"Speedup (Custom/SDPA):    {sdpa_ms / flash_ms:.2f}x")


def test_flash_attn_v4():
    """V4 correctness test: shapes (2,8,512,64), (2,8,1024,64), (2,8,2048,64)."""
    assert HAS_V4, "custom_flash_attn_v4 not built — run: pip install -e ."
    torch.manual_seed(0)
    for N_test in [512, 1024, 2048]:
        q = torch.randn(2, 8, N_test, 64, device='cuda')
        k = torch.randn(2, 8, N_test, 64, device='cuda')
        v = torch.randn(2, 8, N_test, 64, device='cuda')
        ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out, _, _ = custom_flash_attn_v4.flash_attn_v4_forward(q, k, v)
        assert torch.allclose(out, ref, rtol=1e-3, atol=1e-3), \
            f"V4 mismatch at N={N_test}: max_diff={( out - ref).abs().max():.6f}"
        print(f"  V4 N={N_test}: max_diff={(out - ref).abs().max():.6f}  ✓")


if __name__ == '__main__':
    if HAS_V4:
        print("\n── V4 correctness ──")
        test_flash_attn_v4()
