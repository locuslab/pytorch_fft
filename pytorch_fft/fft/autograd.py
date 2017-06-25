import torch
from .fft import fft,ifft,fft2,ifft2,fft3,ifft3


class Fft(torch.autograd.Function):
    def forward(self, X_re, X_im):
        return fft(X_re, X_im)

    def backward(self, grad_output_re, grad_output_im):
        N = grad_output_re.size(-1)
        gr, gi = ifft(grad_output_re,-grad_output_im)
        gr, gi = gr * N, -gi * N
        return gr,gi


class Ifft(torch.autograd.Function):

    def forward(self, k_re, k_im):
        return ifft(k_re, k_im)

    def backward(self, grad_output_re, grad_output_im):
        gr, gi = fft(grad_output_re,-grad_output_im)
        gr, gi = gr, -gi
        return gr, gi


class Fft2d(torch.autograd.Function):
    def forward(self, X_re, X_im):
        return fft2(X_re, X_im)

    def backward(self, grad_output_re, grad_output_im):
        N = grad_output_re.size(-1) * grad_output_re.size(-2)
        gr, gi = ifft2(grad_output_re,-grad_output_im)
        gr, gi = gr * N, -gi * N
        return gr,gi


class Ifft2d(torch.autograd.Function):

    def forward(self, k_re, k_im):
        return ifft2(k_re, k_im)

    def backward(self, grad_output_re, grad_output_im):
        gr, gi = fft2(grad_output_re,-grad_output_im)
        gr, gi = gr, -gi
        return gr, gi


class Fft3d(torch.autograd.Function):
    def forward(self, X_re, X_im):
        return fft3(X_re, X_im)

    def backward(self, grad_output_re, grad_output_im):
        N = grad_output_re.size(-1) * grad_output_re.size(-2) * grad_output_re.size(-3)
        gr, gi = ifft3(grad_output_re,-grad_output_im)
        gr, gi = gr * N, -gi * N
        return gr,gi


class Ifft3d(torch.autograd.Function):

    def forward(self, k_re, k_im):
        return ifft3(k_re, k_im)

    def backward(self, grad_output_re, grad_output_im):
        gr, gi = fft3(grad_output_re,-grad_output_im)
        gr, gi = gr, -gi
        return gr, gi
