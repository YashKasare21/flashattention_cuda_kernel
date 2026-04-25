import torch
import torch.nn.functional as F
import time
import custom_flash_attn

N, D = 1024, 64
ITERS = 20

torch.manual_seed(42)
Q = torch.randn(N, D, dtype=torch.float32, device='cuda')
K = torch.randn(N, D, dtype=torch.float32, device='cuda')
V = torch.randn(N, D, dtype=torch.float32, device='cuda')

# SDPA expects at least 3D input: [batch, seq, dim]
Q3 = Q.unsqueeze(0)
K3 = K.unsqueeze(0)
V3 = V.unsqueeze(0)

# Warmup
for _ in range(5):
    _ = custom_flash_attn.flash_attn_forward(Q, K, V)
    _ = F.scaled_dot_product_attention(Q3, K3, V3)
torch.cuda.synchronize()


def bench(fn, iters):
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000 / iters


flash_ms = bench(lambda: custom_flash_attn.flash_attn_forward(Q, K, V), ITERS)
sdpa_ms  = bench(lambda: F.scaled_dot_product_attention(Q3, K3, V3), ITERS)

# Correctness check
O_custom = custom_flash_attn.flash_attn_forward(Q, K, V)
O_ref    = F.scaled_dot_product_attention(Q3, K3, V3).squeeze(0)
torch.cuda.synchronize()

match    = torch.allclose(O_custom, O_ref, atol=1e-3, rtol=1e-3)
max_diff = (O_custom - O_ref).abs().max().item()

print(f"Shape: Q/K/V = [{N}, {D}]  |  iterations = {ITERS}")
print(f"Max abs diff vs SDPA reference: {max_diff:.2e}")
print(f"Results match (atol=1e-3):      {match}")
print()
print(f"Custom FlashAttn kernel:  {flash_ms:.3f} ms")
print(f"PyTorch SDPA (cuBLAS):    {sdpa_ms:.3f} ms")
print(f"Speedup (Custom/SDPA):    {sdpa_ms / flash_ms:.2f}x")
