# functions/fft.py
import torch
from .._ext import th_fft

def _fft(X_re, X_im, f, rank):
    if not(X_re.dim() >= rank+1 and X_im.dim() >= rank+1): 
        raise ValueError("Inputs must have at least 3 dimensions.")
    if not(X_re.is_cuda and X_im.is_cuda): 
        raise ValueError("Input must be a CUDA tensor.")
    if not(X_re.is_contiguous() and X_im.is_contiguous()):
        raise ValueError("Input must be contiguous.")

    Y1, Y2 = tuple(X_re.new(*X_re.size()).zero_() for _ in range(2))
    f(X_re, X_im, Y1, Y2)
    return (Y1, Y2)

def fft(X_re, X_im): 
    if 'Float' in type(X_re).__name__ :
        f = th_fft.th_Float_fft1
    elif 'Double' in type(X_re).__name__: 
        f = th_fft.th_Double_fft1
    else: 
        raise NotImplementedError
    return _fft(X_re, X_im, f, 1)

def ifft(X_re, X_im): 
    N = X_re.size(-1)
    if 'Float' in type(X_re).__name__ :
        f = th_fft.th_Float_ifft1
    elif 'Double' in type(X_re).__name__: 
        f = th_fft.th_Double_ifft1
    else: 
        raise NotImplementedError   
    Y1, Y2 = _fft(X_re, X_im, f, 1)
    return (Y1/N, Y2/N)

def fft2(X_re, X_im): 
    if 'Float' in type(X_re).__name__ :
        f = th_fft.th_Float_fft2
    elif 'Double' in type(X_re).__name__: 
        f = th_fft.th_Double_fft2
    else: 
        raise NotImplementedError
    return _fft(X_re, X_im, f, 2)

def ifft2(X_re, X_im): 
    N = X_re.size(-1)*X_re.size(-2)
    if 'Float' in type(X_re).__name__ :
        f = th_fft.th_Float_ifft2
    elif 'Double' in type(X_re).__name__: 
        f = th_fft.th_Double_ifft2
    else: 
        raise NotImplementedError   
    Y1, Y2 = _fft(X_re, X_im, f, 2)
    return (Y1/N, Y2/N)

def fft3(X_re, X_im): 
    if 'Float' in type(X_re).__name__ :
        f = th_fft.th_Float_fft3
    elif 'Double' in type(X_re).__name__: 
        f = th_fft.th_Double_fft3
    else: 
        raise NotImplementedError
    return _fft(X_re, X_im, f, 3)

def ifft3(X_re, X_im): 
    N = X_re.size(-1)*X_re.size(-2)*X_re.size(-3)
    if 'Float' in type(X_re).__name__ :
        f = th_fft.th_Float_ifft3
    elif 'Double' in type(X_re).__name__: 
        f = th_fft.th_Double_ifft3
    else: 
        raise NotImplementedError   
    Y1, Y2 = _fft(X_re, X_im, f, 3)
    return (Y1/N, Y2/N)
