import torch
from .fft import fft,ifft,fft2,ifft2,fft3,ifft3

def make_contiguous(*Xs):
    return tuple(X if X.is_contiguous() else X.contiguous() for X in Xs)

class Fft(torch.autograd.Function):
    def forward(self, X_re, X_im):
        X_re, X_im = make_contiguous(X_re, X_im)
        return fft(X_re, X_im)

    def backward(self, grad_output_re, grad_output_im):
        grad_output_re, grad_output_im = make_contiguous(grad_output_re, 
                                                         grad_output_im)
        gi, gr = fft(grad_output_im,grad_output_re)
        return gr,gi


class Ifft(torch.autograd.Function):

    def forward(self, k_re, k_im):
        k_re, k_im = make_contiguous(k_re, k_im)
        return ifft(k_re, k_im)

    def backward(self, grad_output_re, grad_output_im):
        grad_output_re, grad_output_im = make_contiguous(grad_output_re, 
                                                         grad_output_im)
        gi, gr = ifft(grad_output_im,grad_output_re)
        return gr, gi


class Fft2d(torch.autograd.Function):
    def forward(self, X_re, X_im):
        X_re, X_im = make_contiguous(X_re, X_im)
        return fft2(X_re, X_im)

    def backward(self, grad_output_re, grad_output_im):
        grad_output_re, grad_output_im = make_contiguous(grad_output_re, 
                                                         grad_output_im)
        gi, gr = fft2(grad_output_im,grad_output_re)
        return gr,gi


class Ifft2d(torch.autograd.Function):

    def forward(self, k_re, k_im):
        k_re, k_im = make_contiguous(k_re, k_im)
        return ifft2(k_re, k_im)

    def backward(self, grad_output_re, grad_output_im):
        grad_output_re, grad_output_im = make_contiguous(grad_output_re, 
                                                         grad_output_im)
        gi, gr = ifft2(grad_output_im,grad_output_re)
        return gr, gi


class Fft3d(torch.autograd.Function):
    def forward(self, X_re, X_im):
        X_re, X_im = make_contiguous(X_re, X_im)
        return fft3(X_re, X_im)

    def backward(self, grad_output_re, grad_output_im):
        grad_output_re, grad_output_im = make_contiguous(grad_output_re, 
                                                         grad_output_im)
        gi, gr = fft3(grad_output_im,grad_output_re)
        return gr,gi


class Ifft3d(torch.autograd.Function):

    def forward(self, k_re, k_im):
        k_re, k_im = make_contiguous(k_re, k_im)
        return ifft3(k_re, k_im)

    def backward(self, grad_output_re, grad_output_im):
        grad_output_re, grad_output_im = make_contiguous(grad_output_re, 
                                                         grad_output_im)
        gi, gr = ifft3(grad_output_im,grad_output_re)
        return gr, gi
