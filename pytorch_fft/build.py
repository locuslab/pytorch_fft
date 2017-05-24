import os
import torch
from torch.utils.ffi import create_extension

this_file = os.path.dirname(__file__)

sources = ['src/th_fft.c']
headers = ['src/th_fft.h']
defines = []
with_cuda = False

if torch.cuda.is_available():
    print('Including CUDA code.')
    sources += ['src/th_fft_cuda.c']
    headers += ['src/th_fft_cuda.h']
    defines += [('WITH_CUDA', None)]
    with_cuda = True

ffi = create_extension(
    '_ext.th_fft',
    headers=headers,
    sources=sources,
    define_macros=defines,
    relative_to=__file__,
    with_cuda=with_cuda,
    include_dirs=[os.getcwd() + '/src'],
    library_dirs=['/usr/local/cuda/lib64'], 
    libraries=['cufft']
)

if __name__ == '__main__':
    ffi.build()
