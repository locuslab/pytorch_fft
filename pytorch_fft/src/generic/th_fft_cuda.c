#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/th_fft_cuda.c"
#else 

int th_(THCTensor *input1, THCTensor *input2, THCTensor *output1, THCTensor *output2)
{
  // Require that all tensors be of the same size. 
  if (!THCTensor_(isSameSizeAs)(state, input1, output1))
    return 0;
  if (!THCTensor_(isSameSizeAs)(state, input1, output2))
    return 0;
  if (!THCTensor_(isSameSizeAs)(state, input1, input2))
    return 0;

  // Get the tensor dimensions (batchsize, rows, cols). 
  int ndim = THCTensor_(nDimension)(state, input1);
  int batch = 1;
  int i;
  for(i=0; i<ndim-2; i++) {
    batch *= THCTensor_(size)(state, input1, i);
  }
  int r = THCTensor_(size)(state, input1, ndim-2);
  int c = THCTensor_(size)(state, input1, ndim-1);


  // Get actual tensor data.
  real *input1_data = THCTensor_(data)(state, input1);
  real *input2_data = THCTensor_(data)(state, input2);
  real *output1_data = THCTensor_(data)(state, output1);
  real *output2_data = THCTensor_(data)(state, output2);

  // Turn input into a complex array.
  cufft_complex *input_complex; 
  cudaMalloc((void**)&input_complex, sizeof(cufft_complex)*batch*r*c);
  if (cudaGetLastError() != cudaSuccess) {
    fprintf(stderr, "Cuda error: Failed to allocate\n");
    return -1;
  }

  pair2complex(input1_data, input2_data, input_complex, batch*r*c);

  // Allocate the complex array to store the output
  cufft_complex *output_complex; 
  cudaMalloc((void**)&output_complex, sizeof(cufft_complex)*batch*r*c);
  if (cudaGetLastError() != cudaSuccess) {
    fprintf(stderr, "Cuda error: Failed to allocate\n");
    return -1;
  }

  // Make the fft plan. 
  cufftHandle plan;
  int rank = 2;
  int n[2] = {r, c};
  int dist = r*c;
  int nembed[] = {r, c};
  int stride = 1;
  if (cufftPlanMany(&plan, rank, n, 
                    nembed, stride, dist, 
                    nembed, stride, dist, 
                    cufft_type, batch) != CUFFT_SUCCESS) {
    fprintf(stderr, "CUFFT error: Plan creation failed");
    return -1;
  }

  // Execute the fft plan. 
  if (cufft_exec(plan, input_complex, output_complex, cufft_direction) != CUFFT_SUCCESS) {
    fprintf(stderr, "CUFFT error: ExecC2C failed");
    return -1;
  }
  // Not sure if this is necessary. 
  if (cudaThreadSynchronize() != cudaSuccess){
    fprintf(stderr, "Cuda error: Failed to synchronize\n");
    return -1;
  }

  // Copy the real and imaginary parts to the output pointers
  complex2pair(output_complex, output1_data, output2_data, batch*r*c);

  cufftDestroy(plan);
  cudaFree(input_complex);
  cudaFree(output_complex);
  return 1;
}

#endif