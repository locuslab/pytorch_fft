int th_Float_fft1(THCudaTensor *input1, THCudaTensor *input2, THCudaTensor *output1, THCudaTensor *output2);
int th_Float_ifft1(THCudaTensor *input1, THCudaTensor *input2, THCudaTensor *output1, THCudaTensor *output2);
int th_Double_fft1(THCudaDoubleTensor *input1, THCudaDoubleTensor *input2, THCudaDoubleTensor *output1, THCudaDoubleTensor *output2);
int th_Double_ifft1(THCudaDoubleTensor *input1, THCudaDoubleTensor *input2, THCudaDoubleTensor *output1, THCudaDoubleTensor *output2);

int th_Float_fft2(THCudaTensor *input1, THCudaTensor *input2, THCudaTensor *output1, THCudaTensor *output2);
int th_Float_ifft2(THCudaTensor *input1, THCudaTensor *input2, THCudaTensor *output1, THCudaTensor *output2);
int th_Double_fft2(THCudaDoubleTensor *input1, THCudaDoubleTensor *input2, THCudaDoubleTensor *output1, THCudaDoubleTensor *output2);
int th_Double_ifft2(THCudaDoubleTensor *input1, THCudaDoubleTensor *input2, THCudaDoubleTensor *output1, THCudaDoubleTensor *output2);

int th_Float_fft3(THCudaTensor *input1, THCudaTensor *input2, THCudaTensor *output1, THCudaTensor *output2);
int th_Float_ifft3(THCudaTensor *input1, THCudaTensor *input2, THCudaTensor *output1, THCudaTensor *output2);
int th_Double_fft3(THCudaDoubleTensor *input1, THCudaDoubleTensor *input2, THCudaDoubleTensor *output1, THCudaDoubleTensor *output2);
int th_Double_ifft3(THCudaDoubleTensor *input1, THCudaDoubleTensor *input2, THCudaDoubleTensor *output1, THCudaDoubleTensor *output2);

int th_Float_rfft1(THCudaTensor *input1, THCudaTensor *output1, THCudaTensor *output2);
int th_Float_irfft1(THCudaTensor *input1, THCudaTensor *input2, THCudaTensor *output1);
int th_Double_rfft1(THCudaDoubleTensor *input1, THCudaDoubleTensor *output1, THCudaDoubleTensor *output2);
int th_Double_irfft1(THCudaDoubleTensor *input1, THCudaDoubleTensor *input2, THCudaDoubleTensor *output1);

int th_Float_rfft2(THCudaTensor *input1, THCudaTensor *output1, THCudaTensor *output2);
int th_Float_irfft2(THCudaTensor *input1, THCudaTensor *input2, THCudaTensor *output1);
int th_Double_rfft2(THCudaDoubleTensor *input1, THCudaDoubleTensor *output1, THCudaDoubleTensor *output2);
int th_Double_irfft2(THCudaDoubleTensor *input1, THCudaDoubleTensor *input2, THCudaDoubleTensor *output1);

int th_Float_rfft3(THCudaTensor *input1, THCudaTensor *output1, THCudaTensor *output2);
int th_Float_irfft3(THCudaTensor *input1, THCudaTensor *input2, THCudaTensor *output1);
int th_Double_rfft3(THCudaDoubleTensor *input1, THCudaDoubleTensor *output1, THCudaDoubleTensor *output2);
int th_Double_irfft3(THCudaDoubleTensor *input1, THCudaDoubleTensor *input2, THCudaDoubleTensor *output1);