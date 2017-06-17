// Generate Double FFTs
#define cufft_complex cufftDoubleComplex
#define cufft_type CUFFT_Z2Z
#define cufft_exec cufftExecZ2Z

#define cufft_direction CUFFT_FORWARD
#define func_name TH_CONCAT_2(fft, cufft_rank)

#include "generic/th_fft_cuda.c"
#include "THCGenerateDoubleType.h"

#undef cufft_direction
#undef func_name

#define cufft_direction CUFFT_INVERSE
#define func_name TH_CONCAT_2(ifft, cufft_rank)

#include "generic/th_fft_cuda.c"
#include "THCGenerateDoubleType.h"

#undef cufft_direction
#undef func_name

#undef cufft_complex
#undef cufft_type
#undef cufft_exec