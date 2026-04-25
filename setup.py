from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='vector_add_cuda',
    ext_modules=[
        CUDAExtension('vector_add_cuda', [
            'vector_add.cu',
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
