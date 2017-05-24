import torch
torch.manual_seed(0)
# from _ext import th_fft
import functions.fft as cfft
import numpy as np
import numpy.fft as nfft
n = 5
x = torch.randn(n*n).view(1, n, n).double().cuda()

print(x)
print("*"*80)
if torch.cuda.is_available():
    y1, y2 = cfft.fft2(x)
    x_np = x.cpu().numpy().squeeze()
    y_np = nfft.fft2(x_np)
    print(y_np.real)
    print(y1)
    print(y_np.imag)
    print(y2)
    print(np.abs((y1.cpu().numpy()- y_np.real)))
    print(np.abs((y2.cpu().numpy()- y_np.imag)))


