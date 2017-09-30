# A PyTorch wrapper for CUDA FFTs [![License][license-image]][license]

[license-image]: http://img.shields.io/badge/license-Apache--2-blue.svg?style=flat
[license]: LICENSE

*A package that provides a PyTorch C extension for performing batches of 2D CuFFT 
transformations, by [Eric Wong](https://github.com/riceric22)*

## Installation

This package is on PyPi. Install with `pip install pytorch-fft`. 

## Usage

+ From the `pytorch_fft.fft` module, you can use the following to do 
foward and backward FFT transformations (complex to complex)
  + `fft` and `ifft` for 1D transformations
  + `fft2` and `ifft2` for 2D transformations
  + `fft3` and `ifft3` for 3D transformations
+ From the same module, you can also use the following for 
real to complex / complex to real FFT transformations
  + `rfft` and `irfft` for 1D transformations
  + `rfft2` and `irfft2` for 2D transformations
  + `rfft3` and `irfft3` for 3D transformations
+ For an `d`-D transformation, the input tensors are required to have >= (d+1)
  dimensions (n1 x ... x nk x m1 x ... x md) where `n1 x ... x nk` is the
  batch of FFT transformations, and `m1 x ... x md` are the dimensions of the
  `d`-D transformation. `d` must be a number from 1 to 3.
+ Finally, the module contains the following helper functions you may find
useful
  + `reverse(X, group_size=1)` reverses the elements of a tensor and returns
    the result in a new tensor. Note that PyTorch does not current support
    negative slicing, see this
    [issue](https://github.com/pytorch/pytorch/issues/229). If a group size is
    supplied, the elements will be reversed in groups of that size.
  + `expand(X, imag=False, odd=True)` takes a tensor output of a real 2D or 3D
    FFT and expands it with its redundant entries to match the output of a
    complex FFT.
+ For autograd support, use the following functions in the
`pytorch_fft.fft.autograd` module: 
  + `Fft` and `Ifft` for 1D transformations
  + `Fft2d` and `Ifft2d` for 2D transformations
  + `Fft3d` and `Ifft3d` for 3D transformations


```Python
# Example that does a batch of three 2D transformations of size 4 by 5. 
import torch
import pytorch_fft.fft as fft

A_real, A_imag = torch.randn(3,4,5).cuda(), torch.zeros(3,4,5).cuda()
B_real, B_imag = fft.fft2(A_real, A_imag)
fft.ifft2(B_real, B_imag) # equals (A, zeros)

B_real, B_imag = fft.rfft2(A) # is a truncated version which omits
                                   # redundant entries

reverse(torch.arange(0,6)) # outputs [5,4,3,2,1,0]
reverse(torch.arange(0,6), 2) # outputs [4,5,2,3,0,1]

expand(B_real) # is equivalent to  fft.fft2(A, zeros)[0]
expand(B_imag, imag=True) # is equivalent to  fft.fft2(A, zeros)[1]
```


```Python
# Example that uses the autograd for 2D fft:
import torch
from torch.autograd import Variable
import pytorch_fft.fft.autograd as fft
import numpy as np

f = fft.Fft2d()
invf= fft.Ifft2d()

fx, fy = (Variable(torch.arange(0,100).view((1,1,10,10)).cuda(), requires_grad=True), 
          Variable(torch.zeros(1, 1, 10, 10).cuda(),requires_grad=True))
k1,k2 = f(fx,fy)
z = k1.sum() + k2.sum()
z.backward()
print(fx.grad, fy.grad)
```

## Notes
+ This follows NumPy semantics and behavior, so `ifft2(fft2(x)) = x`. Note
  that CuFFT semantics for inverse FFT only flip the sign of the transform,
  but it is not a true inverse.
+ Similarly, the real to complex / complex to real variants also follow NumPy
  semantics and behavior. In the 1D case, this means that for an input of size
  `N`, it returns an output of size `N//2+1` (it omits redundant entries, see
  the [Numpy docs](https://docs.scipy.org/doc/numpy/reference/generated/numpy.fft.rfft.html))
+ The functions in the `pytorch_fft.fft` module do not implement the PyTorch
  autograd `Function`, and are semantically and functionally like their numpy
  equivalents.
+ Autograd functionality is in the `pytorch_fft.fft.autograd` module.

## Repository contents
- pytorch_fft/src: C source code
- pytorch_fft/fft: Python convenience wrapper
- build.py: compilation file
- test.py: tests against NumPy FFTs and Autograd checks

## Issues and Contributions

If you have any issues or feature requests, 
[file an issue](https://github.com/bamos/block/issues)
or [send in a PR](https://github.com/bamos/block/pulls). 

