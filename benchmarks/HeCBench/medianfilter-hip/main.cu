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

#include <hip/hip_runtime.h>
#include "shrUtils.h"
#include "MedianFilter.cu"

#ifndef min
#define min(a,b) (a < b ? a : b)
#endif

// Import host computation function 
extern "C" void MedianFilterHost(unsigned int* uiInputImage, unsigned int* uiOutputImage, 
                                 unsigned int uiWidth, unsigned int uiHeight);

void MedianFilterGPU(
    unsigned int* uiInputImage, 
    unsigned int* uiOutputImage, 
    uchar4* cmDevBufIn,
    unsigned int* cmDevBufOut,
    const int uiImageWidth,
    const int uiImageHeight);

int main(int argc, char** argv)
{
  // Image data file
  const char* cPathAndName = argv[1]; 
  unsigned int uiImageWidth = 1920;   // Image width
  unsigned int uiImageHeight = 1080;  // Image height

  size_t szBuffBytes;                 // Size of main image buffers
  size_t szBuffWords;                 

  //char* cPathAndName = NULL;          // var for full paths to data, src, etc.
  unsigned int* uiInput;              // Host input buffer 
  unsigned int* uiOutput;             // Host output buffer

  // One device processes the whole image
  szBuffWords = uiImageHeight * uiImageWidth;
  szBuffBytes = szBuffWords * sizeof (unsigned int);

  uiInput = (unsigned int*) malloc (szBuffBytes);
  uiOutput = (unsigned int*) malloc (szBuffBytes);

  shrLoadPPM4ub(cPathAndName, (unsigned char **)&uiInput, &uiImageWidth, &uiImageHeight);

  printf("Image File\t = %s\nImage Dimensions = %u w x %u h x %lu bpp\n\n", 
         cPathAndName, uiImageWidth, uiImageHeight, sizeof(unsigned int)<<3);

  uchar4* cmDevBufIn;
  hipMalloc((void**)&cmDevBufIn, szBuffBytes);

  unsigned int* cmDevBufOut;
  hipMalloc((void**)&cmDevBufOut, szBuffBytes);

  // Warmup call 
  MedianFilterGPU (uiInput, uiOutput, cmDevBufIn, 
                   cmDevBufOut, uiImageWidth, uiImageHeight);

  // Process n loops on the GPU
  const int iCycles = 150;
  printf("\nRunning MedianFilterGPU for %d cycles...\n\n", iCycles);
  for (int i = 0; i < iCycles; i++)
  {
    MedianFilterGPU (uiInput, uiOutput, cmDevBufIn, 
                     cmDevBufOut, uiImageWidth, uiImageHeight);
  }

  // Compute on host 
  unsigned int* uiGolden = (unsigned int*)malloc(szBuffBytes);
  MedianFilterHost(uiInput, uiGolden, uiImageWidth, uiImageHeight);

  // Compare GPU and Host results:  Allow variance of 1 GV in up to 0.01% of pixels 
  printf("Comparing GPU Result to CPU Result...\n"); 
  shrBOOL bMatch = shrCompareuit(uiGolden, uiOutput, (uiImageWidth * uiImageHeight), 1.0f, 0.0001f);
  printf("\nGPU Result %s CPU Result within tolerance...\n", 
         (bMatch == shrTRUE) ? "matches" : "DOESN'T match"); 

  // Cleanup and exit
  free(uiGolden);
  free(uiInput);
  free(uiOutput);
  hipFree(cmDevBufIn);
  hipFree(cmDevBufOut);

  if(bMatch == shrTRUE) 
    printf("PASS\n");
  else
    printf("FAIL\n");

  return EXIT_SUCCESS;
}

// Copies input data from host buf to the device, runs kernel, 
// copies output data back to output host buf
void MedianFilterGPU(
    unsigned int* uiInputImage, 
    unsigned int* uiOutputImage, 
    uchar4* cmDevBufIn,
    unsigned int* cmDevBufOut,
    const int uiImageWidth,
    const int uiImageHeight)
{
  size_t szGlobalWorkSize[2];         // 2D global work items (ND range) for Median kernel
  size_t szLocalWorkSize[2];          // 2D local work items (work group) for Median kernel
  const int iBlockDimX = 16;
  const int iBlockDimY = 4;
  const int iLocalPixPitch = iBlockDimX + 2;

  hipMemcpy(cmDevBufIn, (uchar4*)uiInputImage, 
    uiImageWidth * uiImageHeight * sizeof(uchar4), hipMemcpyHostToDevice);

  szLocalWorkSize[0] = iBlockDimX;
  szLocalWorkSize[1] = iBlockDimY;
  szGlobalWorkSize[0] = shrRoundUp((int)szLocalWorkSize[0], uiImageWidth); 
  szGlobalWorkSize[1] = shrRoundUp((int)szLocalWorkSize[1], uiImageHeight);

  dim3 lws(szLocalWorkSize[0], szLocalWorkSize[1]);
  dim3 gws(szGlobalWorkSize[0] / szLocalWorkSize[0], 
           szGlobalWorkSize[1] / szLocalWorkSize[1]);

  hipLaunchKernelGGL(ckMedian, dim3(gws), dim3(lws), sizeof(uchar4)*iLocalPixPitch*(iBlockDimY+2), 0, 
       cmDevBufIn, cmDevBufOut, iLocalPixPitch, uiImageWidth, uiImageHeight);

  hipMemcpy((uchar4*)uiOutputImage, cmDevBufOut, 
    uiImageWidth * uiImageHeight * sizeof(uchar4), hipMemcpyDeviceToHost);
}
