import torch
import time
import custom_cuda_matmul

M, N, K = 1024, 1024, 1024

A = torch.randn(M, K, dtype=torch.float32, device='cuda')
B = torch.randn(K, N, dtype=torch.float32, device='cuda')

# Warmup
_ = custom_cuda_matmul.matmul(A, B)
_ = torch.matmul(A, B)
torch.cuda.synchronize()

# Custom kernel timing
start = time.perf_counter()
C_custom = custom_cuda_matmul.matmul(A, B)
torch.cuda.synchronize()
custom_time = time.perf_counter() - start

# PyTorch native timing
start = time.perf_counter()
C_ref = torch.matmul(A, B)
torch.cuda.synchronize()
torch_time = time.perf_counter() - start

match = torch.allclose(C_custom, C_ref, atol=1e-3, rtol=1e-3)
print(f"Results match: {match}")
print(f"Custom kernel:  {custom_time * 1000:.3f} ms")
print(f"torch.matmul:   {torch_time * 1000:.3f} ms")
