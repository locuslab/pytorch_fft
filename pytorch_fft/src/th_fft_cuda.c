#include <THC/THC.h>
#include <cuda.h>
#include <cufft.h>
#include <complex.h>
// this symbol will be resolved automatically from PyTorch libs
extern THCState *state;

#define th_ TH_CONCAT_4(th_, Real, _, func_name)
#define pair2complex TH_CONCAT_2(Real, 2complex) 
#define complex2pair TH_CONCAT_2(complex2, Real) 

#include "th_fft_generate_helpers.h"

#define cufft_rank 1
#include "th_fft_generate_float.h"
#include "th_fft_generate_double.h"
#undef cufft_rank

#define cufft_rank 2
#include "th_fft_generate_float.h"
#include "th_fft_generate_double.h"
#undef cufft_rank

#define cufft_rank 3
#include "th_fft_generate_float.h"
#include "th_fft_generate_double.h"
#undef cufft_rank
