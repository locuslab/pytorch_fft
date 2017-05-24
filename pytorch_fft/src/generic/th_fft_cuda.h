#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/th_fft_cuda.h"
#else

int th_(fft)(THCudaTensor *input, THCudaTensor *output1, THCudaTensor *output2);
int th_(ifft)(THCudaTensor *input1, THCudaTensor *input2, THCudaTensor *output);
#endif