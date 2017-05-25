#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/helpers.c"
#else 

// helper to convert a pair of real arrays into a complex array
void pair2complex(real *a, real *b, cufft_complex *c, int n)
{
  real *c_tmp = (real*)c;
  cudaMemcpy2D(c_tmp, 2*sizeof(real), 
               a, sizeof(real), 
               sizeof(real), n, cudaMemcpyDeviceToDevice);
  cudaMemcpy2D(c_tmp+1, 2*sizeof(real), 
               b, sizeof(real), 
               sizeof(real), n, cudaMemcpyDeviceToDevice);
}

void complex2pair(cufft_complex *a, real *b, real *c, int n)
{
  real *a_tmp = (real*)a; 
  cudaMemcpy2D(b, sizeof(real), 
               a_tmp, 2*sizeof(real), 
               sizeof(real), n, cudaMemcpyDeviceToDevice);
  cudaMemcpy2D(c, sizeof(real), 
               a_tmp+1, 2*sizeof(real), 
               sizeof(real), n, cudaMemcpyDeviceToDevice);
}
#endif