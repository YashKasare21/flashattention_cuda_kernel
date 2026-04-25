import torch
import vector_add_cuda

def test_vector_add():
    size = 1000
    a = torch.randn(size, device='cuda')
    b = torch.randn(size, device='cuda')
    
    # Run our custom CUDA implementation
    c = vector_add_cuda.add(a, b)
    
    # Run PyTorch's implementation for comparison
    expected = a + b
    
    # Check if results are close enough
    if torch.allclose(c, expected):
        print("Success! CUDA implementation matches PyTorch's + operator.")
    else:
        print("Failure! Results do not match.")
        print(f"Max difference: {torch.max(torch.abs(c - expected))}")

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA is not available on this system. Cannot run test.")
    else:
        test_vector_add()
