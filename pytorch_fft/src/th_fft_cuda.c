#include <THC/THC.h>
#include <cuda.h>
#include <cufft.h>
#include <complex.h>
// this symbol will be resolved automatically from PyTorch libs
extern THCState *state;

#define th_ TH_CONCAT_4(th_, Real, _, func_name)
#define pair2complex TH_CONCAT_2(Real, 2complex) 
#define complex2pair TH_CONCAT_2(complex2, Real) 

// Generate float FFTs

#define cufft_complex cufftComplex
#define cufft_type CUFFT_C2C
#define cufft_exec cufftExecC2C

#include "generic/helpers.c"
#include "THCGenerateFloatType.h"

#define cufft_direction CUFFT_FORWARD
#define func_name fft2

#include "generic/th_fft_cuda.c"
#include "THCGenerateFloatType.h"

#undef cufft_direction
#undef func_name

#define cufft_direction CUFFT_INVERSE
#define func_name ifft2

#include "generic/th_fft_cuda.c"
#include "THCGenerateFloatType.h"

#undef cufft_direction
#undef func_name

#undef cufft_complex
#undef cufft_type
#undef cufft_exec

// Generate Double FFTs
#define cufft_complex cufftDoubleComplex
#define cufft_type CUFFT_Z2Z
#define cufft_exec cufftExecZ2Z

#include "generic/helpers.c"
#include "THCGenerateDoubleType.h"

#define cufft_direction CUFFT_FORWARD
#define func_name fft2

#include "generic/th_fft_cuda.c"
#include "THCGenerateDoubleType.h"

#undef cufft_direction
#undef func_name

#define cufft_direction CUFFT_INVERSE
#define func_name ifft2

#include "generic/th_fft_cuda.c"
#include "THCGenerateDoubleType.h"

#undef cufft_direction
#undef func_name

#undef cufft_complex
#undef cufft_type
#undef cufft_exec