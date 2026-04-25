from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='custom_cuda_matmul',
    ext_modules=[
        CUDAExtension(
            name='custom_cuda_matmul',
            sources=['matmul.cu'],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': ['-O3', '--use_fast_math', '-lineinfo'],
            },
        )
    ],
    cmdclass={'build_ext': BuildExtension},
)