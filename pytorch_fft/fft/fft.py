# functions/fft.py
import torch
from .._ext import th_fft

def fft2(X_re, X_im): 
    if not(X_re.dim() >= 3 and X_im.dim() >= 3): 
        raise ValueError("Inputs must have at least 3 dimensions.")
    if not(X_re.is_cuda and X_im.is_cuda): 
        raise ValueError("Input must be a CUDA tensor.")
    if not(X_re.is_contiguous() and X_im.is_contiguous()):
        raise ValueError("Input must be contiguous.")
    if 'Float' in type(X_re).__name__: 
        Y1, Y2 = tuple(torch.zeros(*X_re.size()).cuda() for _ in range(2))
        th_fft.th_Float_fft2(X_re, X_im, Y1, Y2)
        return (Y1, Y2)
    elif 'Double' in type(X_re).__name__: 
        Y1, Y2 = tuple(torch.zeros(*X_re.size()).double().cuda() for _ in range(2))
        th_fft.th_Double_fft2(X_re, X_im, Y1, Y2)
        return (Y1, Y2)
    else:
        raise NotImplementedError

def ifft2(X_re, X_im): 
    if not(X_re.dim() >= 3 and X_im.dim() >= 3): 
        raise ValueError("Inputs must have at least 3 dimensions.")
    if not(X_re.is_cuda and X_im.is_cuda): 
        raise ValueError("Input must be a CUDA tensor.")
    if not(X_re.is_contiguous() and X_im.is_contiguous()):
        raise ValueError("Input must be contiguous.")
    N = X_re.size(-1)*X_re.size(-2)
    if 'Float' in type(X_re).__name__: 
        Y1, Y2 = tuple(torch.zeros(*X_re.size()).cuda() for _ in range(2))
        th_fft.th_Float_ifft2(X_re, X_im, Y1, Y2)
        return (Y1/N, Y2/N)
    elif 'Double' in type(X_re).__name__: 
        Y1, Y2 = tuple(torch.zeros(*X_re.size()).double().cuda() for _ in range(2))
        th_fft.th_Double_ifft2(X_re, X_im, Y1, Y2)
        return (Y1/N, Y2/N)
    else:
        raise NotImplementedError
