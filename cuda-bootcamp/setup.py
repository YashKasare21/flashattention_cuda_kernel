from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='custom_cuda_ops', # The name of our Python package
    ext_modules=[
        CUDAExtension(
            name='custom_cuda_ops', # The name we will use to import it in Python
            sources=['vector_add.cu'],
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)