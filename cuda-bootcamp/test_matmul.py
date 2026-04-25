import torch
import time
import custom_cuda_matmul
import custom_cuda_matmul_tiled

ITERS = 10
M, N, K = 1024, 1024, 1024

A = torch.randn(M, K, dtype=torch.float32, device='cuda')
B = torch.randn(K, N, dtype=torch.float32, device='cuda')

# Warmup
for _ in range(3):
    _ = custom_cuda_matmul.matmul(A, B)
    _ = custom_cuda_matmul_tiled.matmul_tiled(A, B)
    _ = torch.matmul(A, B)
torch.cuda.synchronize()


def bench(fn, iters):
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000 / iters


naive_ms  = bench(lambda: custom_cuda_matmul.matmul(A, B), ITERS)
tiled_ms  = bench(lambda: custom_cuda_matmul_tiled.matmul_tiled(A, B), ITERS)
torch_ms  = bench(lambda: torch.matmul(A, B), ITERS)

# Correctness check against torch.matmul
C_naive = custom_cuda_matmul.matmul(A, B)
C_tiled = custom_cuda_matmul_tiled.matmul_tiled(A, B)
C_ref   = torch.matmul(A, B)
torch.cuda.synchronize()

naive_ok = torch.allclose(C_naive, C_ref, atol=1e-3, rtol=1e-3)
tiled_ok = torch.allclose(C_tiled, C_ref, atol=1e-3, rtol=1e-3)

print(f"Naive kernel correct:  {naive_ok}")
print(f"Tiled kernel correct:  {tiled_ok}")
print()
print(f"Naive kernel:          {naive_ms:.3f} ms")
print(f"Tiled kernel:          {tiled_ms:.3f} ms")
print(f"torch.matmul (cuBLAS): {torch_ms:.3f} ms")
print()
print(f"Speedup (tiled vs naive):        {naive_ms / tiled_ms:.2f}x")
print(f"Speedup (cuBLAS vs tiled):       {tiled_ms / torch_ms:.2f}x  (>1 = cuBLAS faster)")
