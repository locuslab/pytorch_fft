// Generate float and double helpers
#define cufft_complex cufftComplex

#include "generic/helpers.c"
#include "THCGenerateFloatType.h"

#undef cufft_complex

#define cufft_complex cufftDoubleComplex

#include "generic/helpers.c"
#include "THCGenerateDoubleType.h"

#undef cufft_complex