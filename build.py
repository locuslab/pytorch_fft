import os
import torch
from torch.utils.ffi import create_extension

this_file = os.path.dirname(__file__)

sources = []
headers = []
defines = []
with_cuda = False

if torch.cuda.is_available():
    print('Including CUDA code.')
    sources += ['pytorch_fft/src/th_fft_cuda.c']
    headers += ['pytorch_fft/src/th_fft_cuda.h']
    defines += [('WITH_CUDA', None)]
    with_cuda = True

ffi = create_extension(
    'pytorch_fft._ext.th_fft',
    package=True,
    headers=headers,
    sources=sources,
    define_macros=defines,
    relative_to=__file__,
    with_cuda=with_cuda,
    include_dirs=[os.getcwd() + '/pytorch_fft/src'],
    library_dirs=['/usr/local/cuda/lib64'], 
    libraries=['cufft']
)

if __name__ == '__main__':
    ffi.build()
