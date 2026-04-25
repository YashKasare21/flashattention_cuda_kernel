import torch
import custom_cuda_ops # This is OUR library!

# 1. Create two arrays of 10 million random numbers on the GPU
size = 10_000_000
a = torch.rand(size, dtype=torch.float32, device='cuda')
b = torch.rand(size, dtype=torch.float32, device='cuda')

# 2. Run standard PyTorch addition (The Baseline)
print("Running standard PyTorch addition...")
torch_result = a + b

# 3. Run OUR custom CUDA addition
print("Running custom CUDA addition...")
custom_result = custom_cuda_ops.add(a, b)

# 4. Verify the results match exactly
if torch.allclose(torch_result, custom_result):
    print("✅ SUCCESS! Your custom CUDA kernel matches PyTorch perfectly!")
else:
    print("❌ FAILURE! The math doesn't match.")