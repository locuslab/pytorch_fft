#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/th_fft_cuda.c"
#else

int th_(fft2)(THCTensor *input, THCTensor *output1, THCTensor *output2)
{
  printf("Starting fft\n");
  // THCudaStorage_data(); 
  if (!THCTensor_(isSameSizeAs)(state, input, output1))
    return 0;
  if (!THCTensor_(isSameSizeAs)(state, input, output2))
    return 0;

  int n = THCTensor_(size)(state, input, 0);
  int r = THCTensor_(size)(state, input, 1);
  int c = THCTensor_(size)(state, input, 2);

  real *input_data = THCTensor_(data)(state, input);
  real *output1_data = THCTensor_(data)(state, output1);
  real *output2_data = THCTensor_(data)(state, output2);

  cufft_complex *input_complex; 
  cudaMalloc((void**)&input_complex, sizeof(cufft_complex)*r*c);
  cudaMemset(input_complex, 0, sizeof(cufft_complex)*r*c);

  real *input_tmp = (real*)input_complex;
  cudaMemcpy2D(input_tmp, 2*sizeof(real), 
               input_data, sizeof(real), 
               sizeof(real), r*c, cudaMemcpyDeviceToDevice);


  cufft_complex *output_complex; 
  cudaMalloc((void**)&output_complex, sizeof(cufft_complex)*r*c);
  if (cudaGetLastError() != cudaSuccess) {
    fprintf(stderr, "Cuda error: Failed to allocate\n");
    return -1;
  }

  cufftHandle plan;
  if (cufftPlan2d(&plan, r, c, cufft_type) != CUFFT_SUCCESS) {
    fprintf(stderr, "CUFFT error: Plan creation failed");
    return -1;
  }
  if (cufft_exec(plan, input_complex, output_complex, CUFFT_FORWARD) != CUFFT_SUCCESS) {
    fprintf(stderr, "CUFFT error: ExecC2C Forward failed");
    return -1;
  }

  if (cudaThreadSynchronize() != cudaSuccess){
    fprintf(stderr, "Cuda error: Failed to synchronize\n");
    return -1;
  }

  // Reference for working copy
  // cudaMemcpy(output1_data, input_data, 5*sizeof(input_data[0]), cudaMemcpyDeviceToDevice);
  // printf("first element pitch: %d, %d\n", sizeof(float), sizeof(real));
  // cudaMemcpy2D(output2_data, sizeof(real), 
  //            input_data, sizeof(real), 
  //            sizeof(real), n*r*c, cudaMemcpyDeviceToDevice);

  real *output_tmp = (real*)output_complex;
  cudaMemcpy2D(output1_data, sizeof(real), 
             output_tmp, 2*sizeof(real), 
             sizeof(real), n*r*c, cudaMemcpyDeviceToDevice);

  cudaMemcpy2D(output2_data, sizeof(real), 
             output_tmp+1, 2*sizeof(real), 
             sizeof(real), n*r*c, cudaMemcpyDeviceToDevice);

  // size_t t = sizeof(real)*n*r*c;
  // real *cpu = malloc(t);

  // cudaMemcpy(cpu, output_tmp, t, cudaMemcpyDeviceToHost);
  // int i, j;
  // for(i=0; i<r; i++) {
  //   for(j=0; j<c; j++){
  //     printf("%.4f, ", cpu[i*c + j]);
  //   }
  //   printf("\n");
  // }
  // complex *cpu = malloc(sizeof(complex)*n*r*c);

  // cudaMemcpy(cpu, output_complex, sizeof(complex), cudaMemcpyDeviceToHost);
  // int i, j;
  // for(i=0; i<r; i++) {
  //   for(j=0; j<c; j++){
  //     printf("%.2f %+.2fi, ", creal(cpu[i*c + j]), cimag(cpu[i*c + j]));
  //   }
  //   printf("\n");
  // }
  // free(cpu);

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