# A PyTorch wrapper for CUDA FFTs [![License][license-image]][license]

[license-image]: http://img.shields.io/badge/license-Apache--2-blue.svg?style=flat
[license]: LICENSE

*A package that provides a PyTorch C extension for performing batches of 2D CuFFT 
transformations, by [Eric Wong](https://github.com/riceric22)*

## Installation

This package is on PyPi. Install with `pip install pytorch-fft`. 

## Usage

+ From the `pytorch_fft.fft` module, you can use the following to do 
foward and backward FFT transformation
  + `fft` and `ifft` for 1D transformations
  + `fft2` and `ifft2` for 2D transformations
  + `fft3` and `ifft3` for 3D transformations
+ For an `d`-D transformation, the input tensors are required to have >= (d+1) dimensions (n1 x ... x nk x m1 x ... x md) where `n1 x ... x nk` is the batch of FFT transformations, and `m1 x ... x md` are the dimensions of the `d`-D transformation. `d` must be a number from 1 to 3. 

```Python
# Example that does a batch of three 2D transformations of size 4 by 5. 
import torch
import pytorch_fft.fft as fft

A_real, A_imag = torch.randn(3,4,5).cuda(), torch.zeros(3,4,5).cuda()
B_real, B_imag = fft.fft2(A_real, A_imag)
fft.ifft2(B_real, B_imag) # equals (A_real, A_imag)
```

```Python
# Example that uses the autograd for 2D fft:
import torch
from torch.autograd import Variable
import pytorch_fft.fft.autograd as fft
import numpy as np

f = fft.Fft2d()
invf= fft.Ifft2d()

fx, fy = Variable(torch.Tensor(np.arange(100).reshape((1,1,10,10))).cuda(), requires_grad=True), Variable(torch.zeros(1, 1, 10, 10).cuda(),requires_grad=True)
k1,k2 = f(fx,fy)
z = k1.sum() + k2.sum()
z.backward()
print fx.grad, fy.grad
```

## Notes
+ This follows NumPy semantics, so `ifft2(fft2(x)) = x`. Note that CuFFT semantics 
for inverse FFT only flip the sign of the transform, but it is not a true inverse. 
+ This function is *NOT* a PyTorch autograd `Function`, and as a result is not backprop-able. 
What this package allows you to do is call CuFFT on PyTorch Tensors. 
+ The code currently implements only Complex 
to Complex transformations, and not Real to Complex / Complex to Real transformations. 

## Repository contents
- pytorch_fft/src: C source code
- pytorch_fft/fft: Python convenience wrapper
- build.py: compilation file
- test.py: tests against NumPy FFTs

## Issues and Contributions

If you have any issues or feature requests, 
[file an issue](https://github.com/bamos/block/issues)
or [send in a PR](https://github.com/bamos/block/pulls). 

