# functions/fft.py
import torch
from .._ext import th_fft

def _fft(X_re, X_im, f, rank):
    if not(X_re.dim() >= rank+1 and X_im.dim() >= rank+1): 
        raise ValueError("Inputs must have at least {} dimensions.".format(rank+1))
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

_s = slice(None, None, None)

def _rfft(X, f, rank):
    if not(X.dim() >= rank+1): 
        raise ValueError("Input must have at least {} dimensions.".format(rank+1))
    if not(X.is_cuda): 
        raise ValueError("Input must be a CUDA tensor.")
    if not(X.is_contiguous()):
        raise ValueError("Input must be contiguous.")

    new_size = tuple(X.size())[:-1] + (X.size(-1)//2 + 1,)
    # new_size = tuple(X.size())
    Y1, Y2 = tuple(X.new(*new_size).zero_() for _ in range(2))
    f(X, Y1, Y2)
    # i = tuple(_s for _ in range(X.dim()-1)) + (slice(None, X.size(-1)//2 + 1, ),)
    # print(Y1, i)
    # return (Y1[i], Y2[i])
    return (Y1, Y2)

def rfft(X): 
    if 'Float' in type(X).__name__ :
        f = th_fft.th_Float_rfft1
    elif 'Double' in type(X).__name__: 
        f = th_fft.th_Double_rfft1
    else: 
        raise NotImplementedError
    return _rfft(X, f, 1)

def rfft2(X): 
    if 'Float' in type(X).__name__ :
        f = th_fft.th_Float_rfft2
    elif 'Double' in type(X).__name__: 
        f = th_fft.th_Double_rfft2
    else: 
        raise NotImplementedError
    return _rfft(X, f, 2)

def rfft3(X): 
    if 'Float' in type(X).__name__ :
        f = th_fft.th_Float_rfft3
    elif 'Double' in type(X).__name__: 
        f = th_fft.th_Double_rfft3
    else: 
        raise NotImplementedError
    return _rfft(X, f, 3)

def _irfft(X_re, X_im, f, rank):
    if not(X_re.dim() >= rank+1 and X_im.dim() >= rank+1): 
        raise ValueError("Inputs must have at least {} dimensions.".format(rank+1))
    if not(X_re.is_cuda and X_im.is_cuda): 
        raise ValueError("Input must be a CUDA tensor.")
    if not(X_re.is_contiguous() and X_im.is_contiguous()):
        raise ValueError("Input must be contiguous.")

    # raise NotImplementedError
    N = (X_re.size(-1) - 1)*2
    new_size = tuple(X_re.size())[:-1] + (N,)
    Y = X_re.new(*new_size).zero_()
    f(X_re, X_im, Y)
    M = 1
    for i in range(rank): 
        M *= new_size[-(i+1)]
    return Y/M

def irfft(X_re, X_im): 
    if 'Float' in type(X_re).__name__ :
        f = th_fft.th_Float_irfft1
    elif 'Double' in type(X_re).__name__: 
        f = th_fft.th_Double_irfft1
    else: 
        raise NotImplementedError
    return _irfft(X_re, X_im, f, 1)

def irfft2(X_re, X_im): 
    if 'Float' in type(X_re).__name__ :
        f = th_fft.th_Float_irfft2
    elif 'Double' in type(X_re).__name__: 
        f = th_fft.th_Double_irfft2
    else: 
        raise NotImplementedError
    return _irfft(X_re, X_im, f, 2)

def irfft3(X_re, X_im): 
    if 'Float' in type(X_re).__name__ :
        f = th_fft.th_Float_irfft3
    elif 'Double' in type(X_re).__name__: 
        f = th_fft.th_Double_irfft3
    else: 
        raise NotImplementedError
    return _irfft(X_re, X_im, f, 3)