#include <THC/THC.h>
#include <cuda.h>
#include <cufft.h>
#include <complex.h>
// this symbol will be resolved automatically from PyTorch libs
extern THCState *state;

#define th_(NAME) TH_CONCAT_4(th_, Real, _, NAME)

#define cufft_complex cufftComplex
#define cufft_type CUFFT_C2C
#define cufft_exec cufftExecC2C

// I don't know why this isn't taken care of THGenerateFloatType?
// #define CReal Cuda

#include "generic/th_fft_cuda.c"
#include "THCGenerateFloatType.h"

// #undef CReal

#undef cufft_complex
#undef cufft_type
#undef cufft_exec

#define cufft_complex cufftDoubleComplex
#define cufft_type CUFFT_Z2Z
#define cufft_exec cufftExecZ2Z

// #define CReal CudaDouble

#include "generic/th_fft_cuda.c"
#include "THCGenerateDoubleType.h"

// #undef CReal

#undef cufft_complex
#undef cufft_type
#undef cufft_exec