# functions/fft.py
import torch
from .._ext import th_fft

def fft2(X): 
    assert X.dim() == 3
    assert X.is_cuda
    if 'Float' in type(X).__name__: 
        Y1, Y2 = tuple(torch.zeros(*X.size()).cuda() for _ in range(2))
        th_fft.th_Float_fft2(X, Y1, Y2)
        return (Y1, Y2)
    elif 'Double' in type(X).__name__: 
        Y1, Y2 = tuple(torch.zeros(*X.size()).double().cuda() for _ in range(2))
        th_fft.th_Double_fft2(X, Y1, Y2)
        return (Y1, Y2)
    else:
        raise NotImplementedError