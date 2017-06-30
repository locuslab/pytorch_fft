#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/th_rfft_cuda.c"
#else 

int th_(THCTensor *input1, THCTensor *output1, THCTensor *output2)
{
  // Require that all tensors be of the same size. 
  if (!THCTensor_(isSameSizeAs)(state, output1, output2))
    return 0;

  // Get the tensor dimensions (batchsize, rows, cols). 
  int ndim = THCTensor_(nDimension)(state, input1);
  int batch = 1;
  int i, d;
  for(i=0; i<ndim-cufft_rank; i++) {
    batch *= THCTensor_(size)(state, input1, i);
  }

  // array of dimensions for fft of dimension cufft_rank
  int idim_arr[cufft_rank];
  // product of all dimensions
  int idist=1;

  for(i=ndim-cufft_rank; i<ndim; i++){
    d = THCTensor_(size)(state, input1, i);
    idim_arr[i-(ndim-cufft_rank)] = d;
    idist *= d;
  }

  int odim_arr[cufft_rank];
  int odist=1;

  for(i=ndim-cufft_rank; i<ndim; i++){
    d = THCTensor_(size)(state, output1, i);
    odim_arr[i-(ndim-cufft_rank)] = d;
    odist *= d;
  }


  // Get actual tensor data.
  real *input1_data = THCTensor_(data)(state, input1);
  real *output1_data = THCTensor_(data)(state, output1);
  real *output2_data = THCTensor_(data)(state, output2);

  // Allocate the complex array to store the output
  cufft_complex *output_complex; 
  cudaMalloc((void**)&output_complex, sizeof(cufft_complex)*batch*odist);
  if (cudaGetLastError() != cudaSuccess) {
    fprintf(stderr, "Cuda error: Failed to allocate\n");
    return -1;
  
}
  // Make the fft plan. 
  cufftHandle plan;
  int rank = cufft_rank;
  int stride = 1;
  if (cufftPlanMany(&plan, rank, idim_arr, 
                    idim_arr, stride, idist, 
                    odim_arr, stride, odist, 
                    cufft_type, batch) != CUFFT_SUCCESS) {
    fprintf(stderr, "CUFFT error: Plan creation failed");
    return -1;
  }

  // Execute the fft plan. 
  if (cufft_exec(plan, input1_data, output_complex) !=    CUFFT_SUCCESS) {
    fprintf(stderr, "CUFFT error: cufft_exec failed");
    return -1;
  }
  // Not sure if this is necessary. 
  if (cudaThreadSynchronize() != cudaSuccess){
    fprintf(stderr, "Cuda error: Failed to synchronize\n");
    return -1;
  }

  // Copy the real and imaginary parts to the output pointers
  complex2pair(output_complex, output1_data, output2_data, batch*odist);

  cufftDestroy(plan);
  cudaFree(output_complex);
  return 1;
}

#endif