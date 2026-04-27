from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

_flags = {
    'cxx': ['-O3'],
    'nvcc': ['-O3', '--use_fast_math', '-lineinfo'],
}

setup(
    name='custom_flash_attn',
    ext_modules=[
        CUDAExtension(
            name='custom_flash_attn',
            sources=['src/flash_attn_forward.cu'],
            extra_compile_args=_flags,
        ),
        CUDAExtension(
            name='custom_flash_attn_v2',
            sources=['src/flash_attn_v2.cu'],
            extra_compile_args=_flags,
        ),
    ],
    cmdclass={'build_ext': BuildExtension},
)