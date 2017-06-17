// Generate float FFTs
#define cufft_complex cufftComplex
#define cufft_type CUFFT_C2C
#define cufft_exec cufftExecC2C

#define cufft_direction CUFFT_FORWARD
#define func_name TH_CONCAT_2(fft, cufft_rank)

#include "generic/th_fft_cuda.c"
#include "THCGenerateFloatType.h"

#undef cufft_direction
#undef func_name

#define cufft_direction CUFFT_INVERSE
#define func_name TH_CONCAT_2(ifft, cufft_rank)

#include "generic/th_fft_cuda.c"
#include "THCGenerateFloatType.h"

#undef cufft_direction
#undef func_name

#undef cufft_complex
#undef cufft_type
#undef cufft_exec