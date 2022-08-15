#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <hip/hip_runtime.h>
#include "reference.h"

__global__ 
void entropy(
        float *__restrict__ d_entropy,
    const char*__restrict__ d_val, 
    int height, int width)
{
  const int x = threadIdx.x + blockIdx.x * blockDim.x;
  const int y = threadIdx.y + blockIdx.y * blockDim.y;

  // value of matrix element ranges from 0 inclusive to 16 exclusive
  char count[16];
  for (int i = 0; i < 16; i++) count[i] = 0;

  // total number of valid elements
  char total = 0;

  // 5x5 window
  for(int dy = -2; dy <= 2; dy++) {
    for(int dx = -2; dx <= 2; dx++) {
      int xx = x + dx;
      int yy = y + dy;
      if(xx >= 0 && yy >= 0 && yy < height && xx < width) {
        count[d_val[yy * width + xx]]++;
        total++;
      }
    }
  }

  float entropy = 0;
  if (total < 1) {
    total = 1;
  } else {
    for(int k = 0; k < 16; k++) {
      float p = __fdividef((float)count[k], (float)total);
      entropy -= p * log2f(p);
    }
  }

  if(y < height && x < width) d_entropy[y * width + x] = entropy;
}

template<int bsize_x, int bsize_y>
__global__ void entropy_opt(
       float *__restrict__ d_entropy,
  const  char*__restrict__ d_val, 
  const float*__restrict__ d_logTable,
  int m, int n)
{
  __shared__ int sd_count[16][bsize_y*bsize_x];

  const int x = threadIdx.x + blockIdx.x * blockDim.x;
  const int y = threadIdx.y + blockIdx.y * blockDim.y;
  const int idx = threadIdx.y*bsize_x + threadIdx.x;

  for(int i = 0; i < 16;i++) sd_count[i][idx] = 0;

  char total = 0;
  for(int dy = -2; dy <= 2; dy++) {
    for(int dx = -2; dx <= 2; dx++) {
      int xx = x + dx,
          yy = y + dy;

      if(xx >= 0 && yy >= 0 && yy < m && xx < n) {
        sd_count[d_val[yy*n+xx]][idx]++;
        total++;
      }
    }
  }

  float entropy = 0;
  for(int k = 0; k < 16; k++)
    entropy -= d_logTable[sd_count[k][idx]];
  
  entropy = entropy / total + log2f(total);
  if(y < m && x < n) d_entropy[y*n+x] = entropy;
}

int main(int argc, char* argv[]) {
  if (argc != 3) {
    printf("Usage: %s <width> <height>\n", argv[0]);
    return 1;
  }
  const int width = atoi(argv[1]); 
  const int height = atoi(argv[2]); 

  const int input_bytes = width * height * sizeof(char);
  const int output_bytes = width * height * sizeof(float);
  char* input = (char*) malloc (input_bytes);
  float* output = (float*) malloc (output_bytes);
  float* output_ref = (float*) malloc (output_bytes);

  srand(123);
  for (int i = 0; i < height; i++)
    for (int j = 0; j < width; j++)
      input[i * width + j] = rand() % 16;

  char* d_input;
  hipMalloc((void**)&d_input, input_bytes);
  hipMemcpy(d_input, input, input_bytes, hipMemcpyHostToDevice);

  float* d_output;
  hipMalloc((void**)&d_output, output_bytes);

  dim3 grids ((width+15)/16, (height+15)/16);
  dim3 blocks (16, 16);

  // baseline kernel
  for (int i = 0; i < 100; i++)
    hipLaunchKernelGGL(entropy, grids, blocks, 0, 0, d_output, d_input, height, width);

  hipMemcpy(output, d_output, output_bytes, hipMemcpyDeviceToHost);

  // optimized kernel

  float logTable[26];
  for (int i = 0; i <= 25; i++) logTable[i] = i <= 1 ? 0 : i*log2f(i);
  float* d_logTable;
  hipMalloc((void**)&d_logTable, sizeof(logTable));
  hipMemcpy(d_logTable, logTable, sizeof(logTable), hipMemcpyHostToDevice);
 
  for (int i = 0; i < 100; i++)
    hipLaunchKernelGGL(HIP_KERNEL_NAME(entropy_opt<16, 16>), dim3(grids), dim3(blocks ),
      0, 0, d_output, d_input, d_logTable, height, width);
  hipMemcpy(output, d_output, output_bytes, hipMemcpyDeviceToHost);

  hipFree(d_input);
  hipFree(d_output);

  // verify
  reference(output_ref, input, height, width);

  bool ok = true;
  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      if (fabsf(output[i * width + j] - output_ref[i * width + j]) > 1e-3f) {
        ok = false; 
        break;
      }
    }
    if (!ok) break;
  }
  printf("%s\n", ok ? "PASS" : "FAIL");
 
  free(input);
  free(output);
  free(output_ref);
  return 0;
}
