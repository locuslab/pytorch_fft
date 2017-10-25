import torch
from .fft import fft,ifft,fft2,ifft2,fft3,ifft3,rfft,irfft,rfft2,irfft2,rfft3,irfft3

def make_contiguous(*Xs):
    return tuple(X if X.is_contiguous() else X.contiguous() for X in Xs)

def contiguous_clone(X):
    if X.is_contiguous():
        return X.clone()
    else:
        return X.contiguous()

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


class Rfft(torch.autograd.Function):
    def forward(self, X_re):
        X_re = X_re.contiguous()
        self._to_save_input_size = X_re.size(-1)
        return rfft(X_re)

    def backward(self, grad_output_re, grad_output_im):
        # Clone the array and make contiguous if needed
        grad_output_re = contiguous_clone(grad_output_re)
        grad_output_im = contiguous_clone(grad_output_im)

        if self._to_save_input_size & 1:
            grad_output_re[...,1:] /= 2
        else:
            grad_output_re[...,1:-1] /= 2

        if self._to_save_input_size & 1:
            grad_output_im[...,1:] /= 2
        else:
            grad_output_im[...,1:-1] /= 2

        gr = irfft(grad_output_re,grad_output_im,self._to_save_input_size, normalize=False)
        return gr


class Irfft(torch.autograd.Function):

    def forward(self, k_re, k_im):
        k_re, k_im = make_contiguous(k_re, k_im)
        return irfft(k_re, k_im)

    def backward(self, grad_output_re):
        grad_output_re = grad_output_re.contiguous()
        gr, gi = rfft(grad_output_re)

        N = grad_output_re.size(-1)
        gr[...,0] /= N
        gr[...,1:-1] /= N/2
        gr[...,-1] /= N

        gi[...,0] /= N
        gi[...,1:-1] /= N/2
        gi[...,-1] /= N
        return gr, gi


class Rfft2d(torch.autograd.Function):
    def forward(self, X_re):
        X_re = X_re.contiguous()
        self._to_save_input_size = X_re.size(-1)
        return rfft2(X_re)

    def backward(self, grad_output_re, grad_output_im):
        # Clone the array and make contiguous if needed
        grad_output_re = contiguous_clone(grad_output_re)
        grad_output_im = contiguous_clone(grad_output_im)

        if self._to_save_input_size & 1:
            grad_output_re[...,1:] /= 2
        else:
            grad_output_re[...,1:-1] /= 2

        if self._to_save_input_size & 1:
            grad_output_im[...,1:] /= 2
        else:
            grad_output_im[...,1:-1] /= 2

        gr = irfft2(grad_output_re,grad_output_im,self._to_save_input_size, normalize=False)
        return gr


class Irfft2d(torch.autograd.Function):

    def forward(self, k_re, k_im):
        k_re, k_im = make_contiguous(k_re, k_im)
        return irfft2(k_re, k_im)

    def backward(self, grad_output_re):
        grad_output_re = grad_output_re.contiguous()
        gr, gi = rfft2(grad_output_re)

        N = grad_output_re.size(-1) * grad_output_re.size(-2)
        gr[...,0] /= N
        gr[...,1:-1] /= N/2
        gr[...,-1] /= N

        gi[...,0] /= N
        gi[...,1:-1] /= N/2
        gi[...,-1] /= N
        return gr, gi


class Rfft3d(torch.autograd.Function):
    def forward(self, X_re):
        X_re = X_re.contiguous()
        self._to_save_input_size = X_re.size(-1)
        return rfft3(X_re)

    def backward(self, grad_output_re, grad_output_im):
        # Clone the array and make contiguous if needed
        grad_output_re = contiguous_clone(grad_output_re)
        grad_output_im = contiguous_clone(grad_output_im)

        if self._to_save_input_size & 1:
            grad_output_re[...,1:] /= 2
        else:
            grad_output_re[...,1:-1] /= 2

        if self._to_save_input_size & 1:
            grad_output_im[...,1:] /= 2
        else:
            grad_output_im[...,1:-1] /= 2

        gr = irfft3(grad_output_re,grad_output_im,self._to_save_input_size, normalize=False)
        return gr


class Irfft3d(torch.autograd.Function):

    def forward(self, k_re, k_im):
        k_re, k_im = make_contiguous(k_re, k_im)
        return irfft3(k_re, k_im)

    def backward(self, grad_output_re):
        grad_output_re = grad_output_re.contiguous()
        gr, gi = rfft3(grad_output_re)

        N = grad_output_re.size(-1) * grad_output_re.size(-2) * grad_output_re.size(-3)
        gr[...,0] /= N
        gr[...,1:-1] /= N/2
        gr[...,-1] /= N

        gi[...,0] /= N
        gi[...,1:-1] /= N/2
        gi[...,-1] /= N
        return gr, gi

