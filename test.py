import torch
torch.manual_seed(0)
# from _ext import th_fft
import pytorch_fft.fft as cfft
import numpy as np
import numpy.fft as nfft

def run_fft(x, z, _f1, _f2, _if1, _if2, atol):
    y1, y2 = _f1(x, z)
    x_np = x.cpu().numpy().squeeze()
    y_np = _f2(x_np)
    assert np.allclose(y1.cpu().numpy(), y_np.real, atol=atol)
    assert np.allclose(y2.cpu().numpy(), y_np.imag, atol=atol)

    x0, z0 = _if1(y1, y2)
    x0_np = _if2(y_np)
    assert np.allclose(x0.cpu().numpy(), x0_np.real, atol=atol)
    assert np.allclose(z0.cpu().numpy(), x0_np.imag, atol=atol)


def test_acc(_f1, _f2, _if1, _if2): 
    batch = 3
    nch = 4
    n = 5
    m = 7
    x = torch.randn(batch*nch*n*m).view(batch, nch, n, m).cuda()
    z = torch.zeros(batch, nch, n, m).cuda()
    run_fft(x, z, _f1, _f2, _if1, _if2, 1e-6)
    run_fft(x.double(), z.double(), _f1, _f2, _if1, _if2, 1e-14)

if __name__ == "__main__": 
    if torch.cuda.is_available():
        nfft3 = lambda x: nfft.fftn(x,axes=(1,2,3))
        nifft3 = lambda x: nfft.ifftn(x,axes=(1,2,3))

        cfs = [cfft.fft, cfft.fft2, cfft.fft3]
        nfs = [nfft.fft, nfft.fft2, nfft3]
        cifs = [cfft.ifft, cfft.ifft2, cfft.ifft3]
        nifs = [nfft.ifft, nfft.ifft2, nifft3]
        
        for args in zip(cfs, nfs, cifs, nifs):
            test_acc(*args)
    else:
        print("Cuda not available, cannot test.")
