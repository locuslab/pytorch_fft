# A PyTorch wrapper for CUDA FFTs [![License][license-image]][license]

[license-image]: http://img.shields.io/badge/license-Apache--2-blue.svg?style=flat
[license]: LICENSE

*A package that provides a PyTorch C extension for performing batches of 2D CuFFT 
transformations, by [Eric Wong](https://github.com/riceric22)*

## Installation

This package is on PyPi. Install with `pip install pytorch-fft`. 

## Usage

+ From the `pytorch_fft.fft` module, you can use `fft2` and `ifft2` to do the forward
and backward FFT transformations. 
+ The input tensors are required to be 3D tensors of dimension `batch x row x col`
where `batch` is the number of FFT transformations, and `row x col` are the dimensions
of the input to each transformation. 

```Python
import torch
import pytorch_fft.fft as fft

A_real, A_imag = torch.randn(3,4,5).cuda(), torch.zeros(3,4,5).cuda()
B_real, B_imag = fft.fft2(A_real, A_imag)
fft.ifft2(B_real, B_imag) # equals (A_real, A_imag)
```

## Notes
+ This follows NumPy semantics, so `ifft2(fft2(x)) = x`. Note that CuFFT semantics 
for inverse FFT only flip the sign of the transform, but it is not a true inverse. 
+ This function is *NOT* a PyTorch autograd `Function`, and as a result is not backprop-able. 
What this package allows you to do is call CuFFT on PyTorch Tensors. 
+ The code currently only implements batched 2D transformation, for Complex 
to Complex transformations. If you require a different number of dimensions, 
the source code can be easily extended. 

## Repository contents
- pytorch_fft/src: C source code
- pytorch_fft/fft: Python convenience wrapper
- build.py: compilation file
- test.py: tests against NumPy FFTs

## Issues and Contributions

If you have any issues or feature requests, 
[file an issue](https://github.com/bamos/block/issues)
or [send in a PR](https://github.com/bamos/block/pulls). 

