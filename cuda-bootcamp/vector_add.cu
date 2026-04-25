#include <torch/extension.h>

// 1. THIS IS THE CUDA KERNEL (Runs on the GPU)
// __global__ tells the compiler: "This function is called by the CPU, but runs on the GPU."
__global__ void vector_add_cuda_kernel(const float* a, const float* b, float* c, int size) {
    // A GPU has thousands of threads. This line calculates the unique ID for the current thread.
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Make sure we don't try to access memory outside our array size
    if (idx < size) {
        c[idx] = a[idx] + b[idx]; // Every thread does ONE addition simultaneously!
    }
}

// 2. THIS IS THE C++ HELPER FUNCTION (Runs on the CPU)
// It prepares the memory and launches the GPU kernel
torch::Tensor vector_add(torch::Tensor a, torch::Tensor b) {
    // Create an empty tensor to hold the result
    auto c = torch::zeros_like(a);
    
    // We group threads into "blocks". 256 is a standard, safe number of threads per block.
    const int threads_per_block = 256;
    
    // Calculate how many blocks we need to cover the whole array
    const int blocks = (a.numel() + threads_per_block - 1) / threads_per_block;
    
    // LAUNCH THE KERNEL! The <<<blocks, threads>>> syntax is special to CUDA.
    vector_add_cuda_kernel<<<blocks, threads_per_block>>>(
        a.data_ptr<float>(), 
        b.data_ptr<float>(), 
        c.data_ptr<float>(), 
        a.numel()
    );
    
    return c;
}

// 3. THIS IS THE PYBIND11 MODULE
// This exposes our C++ function to Python so we can 'import' it.
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("add", &vector_add, "A simple vector addition using CUDA.");
}