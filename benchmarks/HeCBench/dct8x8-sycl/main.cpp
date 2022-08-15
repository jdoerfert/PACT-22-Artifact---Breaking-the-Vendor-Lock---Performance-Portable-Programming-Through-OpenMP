/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "common.h"
#include "DCT8x8.h"

void DCT8x8(
    queue &q,
    buffer<float, 1> &d_Dst,
    buffer<float, 1> &d_Src,
    unsigned int stride,
    unsigned int imageH,
    unsigned int imageW,
    int dir
);

void Verify(const float* h_OutputGPU, 
                  float* h_OutputCPU, 
            const float* h_Input, 
            const unsigned int stride,
            const unsigned int imageH,
            const unsigned int imageW,
            const int dir )
{
  printf("Comparing against Host/C++ computation...\n"); 
  DCT8x8CPU(h_OutputCPU, h_Input, stride, imageH, imageW, dir);
  double sum = 0, delta = 0;
  double L2norm;
  for(unsigned int i = 0; i < imageH; i++)
    for(unsigned int j = 0; j < imageW; j++){
      sum += h_OutputCPU[i * stride + j] * h_OutputCPU[i * stride + j];
      delta += (h_OutputGPU[i * stride + j] - h_OutputCPU[i * stride + j]) * 
               (h_OutputGPU[i * stride + j] - h_OutputCPU[i * stride + j]);
    }
  L2norm = std::sqrt(delta / sum);
  printf("Relative L2 norm: %.3e\n\n", L2norm);
  if (L2norm < 1E-6) 
    printf("PASS\n"); 
  else
    printf("FAIL\n");
}

////////////////////////////////////////////////////////////////////////////////
// Main program
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
  float *h_Input, *h_OutputCPU, *h_OutputGPU;

  const unsigned int imageW = 2048, imageH = 2048, stride = 2048;


  printf("Allocating and initializing host memory...\n");
  h_Input     = (float *)malloc(imageH * stride * sizeof(float));
  h_OutputCPU = (float *)malloc(imageH * stride * sizeof(float));
  h_OutputGPU = (float *)malloc(imageH * stride * sizeof(float));
  srand(2009);
  for(unsigned int i = 0; i < imageH; i++)
    for(unsigned int j = 0; j < imageW; j++)
      h_Input[i * stride + j] = (float)rand() / (float)RAND_MAX;

  const int numIterations = 150;

#ifdef USE_GPU
  gpu_selector dev_sel;
#else
  cpu_selector dev_sel;
#endif
  queue q(dev_sel);

  buffer<float, 1> d_Input (h_Input, imageH * stride);
  buffer<float, 1> d_Output (imageH * stride);

  int dir = DCT_FORWARD;
  printf("Performing Forward DCT8x8 of %u x %u image on the device\n\n", imageH, imageW);

  for(int iter = 0; iter < numIterations; iter++)
    DCT8x8(
        q,
        d_Output,
        d_Input,
        stride,
        imageH,
        imageW,
        dir );

  q.submit([&] (handler &cgh) {
    auto acc = d_Output.get_access<sycl_read>(cgh);
    cgh.copy(acc, h_OutputGPU); 
  });
  q.wait();

  Verify(h_OutputGPU, h_OutputCPU, h_Input, stride, imageH, imageW, dir);

  dir = DCT_INVERSE;
  printf("Performing Inverse DCT8x8 of %u x %u image on the device\n\n", imageH, imageW);

  for(int iter = 0; iter < numIterations; iter++)
    DCT8x8(
        q,
        d_Output,
        d_Input,
        stride,
        imageH,
        imageW,
        dir );

  q.submit([&] (handler &cgh) {
    auto acc = d_Output.get_access<sycl_read>(cgh);
    cgh.copy(acc, h_OutputGPU); 
  });
  q.wait();

  Verify(h_OutputGPU, h_OutputCPU, h_Input, stride, imageH, imageW, dir);

  free(h_OutputGPU);
  free(h_OutputCPU);
  free(h_Input);

  return 0;
}
