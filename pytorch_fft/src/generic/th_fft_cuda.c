#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/th_fft_cuda.c"
#else

int th_(fft2)(THCTensor *input, THCTensor *output1, THCTensor *output2)
{
  // Require that all tensors be of the same size. 
  if (!THCTensor_(isSameSizeAs)(state, input, output1))
    return 0;
  if (!THCTensor_(isSameSizeAs)(state, input, output2))
    return 0;

  // Get the tensor dimensions (batchsize, rows, cols). 
  int batch = THCTensor_(size)(state, input, 0);
  int r = THCTensor_(size)(state, input, 1);
  int c = THCTensor_(size)(state, input, 2);


  // Get actual tensor data.
  real *input_data = THCTensor_(data)(state, input);
  real *output1_data = THCTensor_(data)(state, output1);
  real *output2_data = THCTensor_(data)(state, output2);

  // Turn input into a complex array with zero imaginary part. 
  cufft_complex *input_complex; 
  cudaMalloc((void**)&input_complex, sizeof(cufft_complex)*batch*r*c);
  if (cudaGetLastError() != cudaSuccess) {
    fprintf(stderr, "Cuda error: Failed to allocate\n");
    return -1;
  }
  
  cudaMemset(input_complex, 0, sizeof(cufft_complex)*batch*r*c);

  real *input_tmp = (real*)input_complex;
  cudaMemcpy2D(input_tmp, 2*sizeof(real), 
               input_data, sizeof(real), 
               sizeof(real), batch*r*c, cudaMemcpyDeviceToDevice);

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
  int idist = r*c;
  int odist = r*c;
  int inembed[] = {r, c};
  int onembed[] = {r, c};
  int istride = 1;
  int ostride = 1; 
  if (cufftPlanMany(&plan, rank, n, 
                    inembed, istride, idist, 
                    onembed, ostride, odist, 
                    cufft_type, batch) != CUFFT_SUCCESS) {
    fprintf(stderr, "CUFFT error: Plan creation failed");
    return -1;
  }

  // Execute the fft plan. 
  if (cufft_exec(plan, input_complex, output_complex, CUFFT_FORWARD) != CUFFT_SUCCESS) {
    fprintf(stderr, "CUFFT error: ExecC2C Forward failed");
    return -1;
  }
  // Not sure if this is necessary. 
  if (cudaThreadSynchronize() != cudaSuccess){
    fprintf(stderr, "Cuda error: Failed to synchronize\n");
    return -1;
  }

  // Copy the real and imaginary parts to the output pointers
  real *output_tmp = (real*)output_complex;
  cudaMemcpy2D(output1_data, sizeof(real), 
               output_tmp, 2*sizeof(real), 
               sizeof(real), batch*r*c, cudaMemcpyDeviceToDevice);

  cudaMemcpy2D(output2_data, sizeof(real), 
             output_tmp+1, 2*sizeof(real), 
             sizeof(real), batch*r*c, cudaMemcpyDeviceToDevice);

  cufftDestroy(plan);
  cudaFree(input_complex);
  cudaFree(output_complex);
  return 1;
}

int th_(ifft2)(THCTensor *input1, THCTensor *input2, THCTensor *output)
{
  // THCTensor_resizeAs(state, grad_input, grad_output);
  // THCTensor_fill(state, grad_input, 1);
  return 1;
}

#endif