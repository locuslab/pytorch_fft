// Generate float FFTs
#define cufft_complex cufftComplex

#define cufft_type CUFFT_C2C
#define cufft_exec cufftExecC2C

#define cufft_direction CUFFT_FORWARD
#define func_name TH_CONCAT_2(fft, cufft_rank)

#include "generic/th_fft_cuda.c"
#include "THCGenerateFloatType.h"

#undef func_name
#undef cufft_direction

#define cufft_direction CUFFT_INVERSE
#define func_name TH_CONCAT_2(ifft, cufft_rank)

#include "generic/th_fft_cuda.c"
#include "THCGenerateFloatType.h"

#undef func_name
#undef cufft_direction


#undef cufft_type
#undef cufft_exec

// Generate float rFFTs
#define cufft_type CUFFT_R2C
#define cufft_exec cufftExecR2C
#define func_name TH_CONCAT_2(rfft, cufft_rank)

#include "generic/th_rfft_cuda.c"
#include "THCGenerateFloatType.h"

#undef func_name
#undef cufft_type
#undef cufft_exec

#define cufft_type CUFFT_C2R
#define cufft_exec cufftExecC2R
#define func_name TH_CONCAT_2(irfft, cufft_rank)

#include "generic/th_irfft_cuda.c"
#include "THCGenerateFloatType.h"

#undef func_name
#undef cufft_type
#undef cufft_exec

#undef cufft_complex