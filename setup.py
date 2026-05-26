from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

_sm75 = {
    'cxx': ['-O3'],
    'nvcc': ['-O3', '--use_fast_math', '-lineinfo', '-arch=sm_75'],
}
_generic = {
    'cxx': ['-O3'],
    'nvcc': ['-O3', '--use_fast_math', '-lineinfo'],
}

setup(
    name='flash_attn_cuda',
    ext_modules=[
        CUDAExtension('custom_flash_attn',    ['src/flash_attn_v1.cu'],           extra_compile_args=_generic),
        CUDAExtension('custom_flash_attn_v2', ['src/flash_attn_v2.cu'],           extra_compile_args=_generic),
        CUDAExtension('custom_flash_attn_v3', ['src/flash_attn_v3.cu'],           extra_compile_args=_sm75),
        CUDAExtension('custom_flash_attn_v4', ['src/flash_attn_v4.cu'],           extra_compile_args=_sm75),
        CUDAExtension('custom_flash_attn_v5', ['src/flash_attn_backward_v5.cu'], extra_compile_args=_sm75),
    ],
    cmdclass={'build_ext': BuildExtension},
)
