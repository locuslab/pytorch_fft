import torch
torch.manual_seed(0)
# from _ext import th_fft
import pytorch_fft.fft as cfft
import numpy as np
import numpy.fft as nfft

def run_fft(x, z):
    if torch.cuda.is_available():
        y1, y2 = cfft.fft2(x, z)
        x_np = x.cpu().numpy().squeeze()
        y_np = np.zeros(tuple(x.size()), dtype=np.complex)
        for i in range(x.size(0)): 
            y_np[i] = nfft.fft2(x_np[i])

        assert np.allclose(y1.cpu().numpy(), y_np.real)
        assert np.allclose(y2.cpu().numpy(), y_np.imag)

        x0, z0 = cfft.ifft2(y1, y2)
        x0_np = np.zeros(tuple(x.size()), dtype=np.complex)
        for i in range(x.size(0)): 
            x0_np[i] = nfft.ifft2(y_np[i])

        assert np.allclose(x0.cpu().numpy(), x0_np.real)
        assert np.allclose(z0.cpu().numpy(), x0_np.imag)
    else:
        print("Cuda not available, cannot test.")

def test_acc(): 
    batch = 3
    n = 5
    m = 7
    x = torch.randn(batch*n*m).view(batch, n, m).cuda()
    z = torch.zeros(batch, n, m).cuda()
    run_fft(x, z)
    run_fft(x.double(), z.double())

if __name__ == "__main__": 
    test_acc()
