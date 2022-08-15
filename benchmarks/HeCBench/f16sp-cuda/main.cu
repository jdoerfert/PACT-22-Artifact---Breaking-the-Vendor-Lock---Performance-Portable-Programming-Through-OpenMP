/**
 * Copyright 2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <cuda_fp16.h>

#define NUM_OF_BLOCKS 1024
#define NUM_OF_THREADS 128

__forceinline__ __device__ 
void reduceInShared_intrinsics(half2 * const v)
{
    int lid = threadIdx.x;	
    if(lid<64) v[lid] = __hadd2( v[lid], v[lid+64]);
    __syncthreads();
    if(lid<32) v[lid] = __hadd2( v[lid], v[lid+32]);
    __syncthreads();
    if(lid<32) v[lid] = __hadd2( v[lid], v[lid+16]);
    __syncthreads();
    if(lid<32) v[lid] = __hadd2( v[lid], v[lid+8]);
    __syncthreads();
    if(lid<32) v[lid] = __hadd2( v[lid], v[lid+4]);
    __syncthreads();
    if(lid<32) v[lid] = __hadd2( v[lid], v[lid+2]);
    __syncthreads();
    if(lid<32) v[lid] = __hadd2( v[lid], v[lid+1]);
    __syncthreads();
}

__forceinline__ __device__
void reduceInShared_native(half2 * const v)
{
    int lid = threadIdx.x;	
    if(lid<64) v[lid] = v[lid] + v[lid+64];
    __syncthreads();
    if(lid<32) v[lid] = v[lid] + v[lid+32];
    __syncthreads();
    if(lid<32) v[lid] = v[lid] + v[lid+16];
    __syncthreads();
    if(lid<32) v[lid] = v[lid] + v[lid+8];
    __syncthreads();
    if(lid<32) v[lid] = v[lid] + v[lid+4];
    __syncthreads();
    if(lid<32) v[lid] = v[lid] + v[lid+2];
    __syncthreads();
    if(lid<32) v[lid] = v[lid] + v[lid+1];
    __syncthreads();
}

__global__
void scalarProductKernel_intrinsics(
        half2 const * const a,
        half2 const * const b,
        float * const results,
        size_t const size
        )
{
    const int stride = gridDim.x*blockDim.x;
    __shared__ half2 shArray[NUM_OF_THREADS];

    half2 value = __float2half2_rn(0.f);

    for (int i = threadIdx.x + blockDim.x * blockIdx.x; i < size; i+=stride)
    {
        value = __hfma2(a[i], b[i], value);
    }

    shArray[threadIdx.x] = value;
    __syncthreads();
    reduceInShared_intrinsics(shArray);

    if (threadIdx.x == 0)
    {
        half2 result = shArray[0];
        float f_result = __low2float(result) + __high2float(result);
        results[blockIdx.x] = f_result;
    }
}

__global__
void scalarProductKernel_native(
        half2 const * const a,
        half2 const * const b,
        float * const results,
        size_t const size
        )
{
    const int stride = gridDim.x*blockDim.x;
    __shared__ half2 shArray[NUM_OF_THREADS];

    half2 value(0.f, 0.f);
    shArray[threadIdx.x] = value;

    for (int i = threadIdx.x + blockDim.x * blockIdx.x; i < size; i+=stride)
    {
        value = a[i] * b[i] + value;
    }

    shArray[threadIdx.x] = value;
    __syncthreads();
    reduceInShared_native(shArray);

    if (threadIdx.x == 0)
    {
        half2 result = shArray[0];
        float f_result = (float)result.y + (float)result.x;
        results[blockIdx.x] = f_result;
    }
}

void generateInput(half2 * a, size_t size)
{
    for (size_t i = 0; i < size; ++i)
    {
        half2 temp;
        temp.x = static_cast<float>(rand() % 4);
        temp.y = static_cast<float>(rand() % 2);
        a[i] = temp;
    }
}

int main(int argc, char *argv[])
{
    size_t size = NUM_OF_BLOCKS*NUM_OF_THREADS*16;

    half2 * a, *b;
    half2 * d_a, *d_b;

    float * r;  // result
    float * d_r;

    a = (half2*) malloc (size*sizeof(half2));
    b = (half2*) malloc (size*sizeof(half2));
    cudaMalloc((void**)&d_a, size*sizeof(half2));
    cudaMalloc((void**)&d_b, size*sizeof(half2));

    r = (float*) malloc (NUM_OF_BLOCKS*sizeof(float));
    cudaMalloc((void**)&d_r, NUM_OF_BLOCKS*sizeof(float));

    srand(123); 
    generateInput(a, size);
    cudaMemcpy(d_a, a, size*sizeof(half2), cudaMemcpyHostToDevice);

    generateInput(b, size);
    cudaMemcpy(d_b, b, size*sizeof(half2), cudaMemcpyHostToDevice);

    for (int i = 0; i < 10000; i++)
      scalarProductKernel_intrinsics<<<NUM_OF_BLOCKS, NUM_OF_THREADS>>>(d_a, d_b, d_r, size);

    cudaMemcpy(r, d_r, NUM_OF_BLOCKS*sizeof(float), cudaMemcpyDeviceToHost);


    float result_intrinsics = 0;
    for (int i = 0; i < NUM_OF_BLOCKS; ++i)
    {
        result_intrinsics += r[i];
    }
    printf("Result intrinsics\t: %f \n", result_intrinsics);

    for (int i = 0; i < 10000; i++)
      scalarProductKernel_native<<<NUM_OF_BLOCKS, NUM_OF_THREADS>>>(d_a, d_b, d_r, size);

    cudaMemcpy(r, d_r, NUM_OF_BLOCKS*sizeof(float), cudaMemcpyDeviceToHost);

    float result_native = 0;
    for (int i = 0; i < NUM_OF_BLOCKS; ++i)
    {
        result_native += r[i];
    }
    printf("Result native operators\t: %f \n", result_native);


    printf("fp16ScalarProduct %s\n", (fabs(result_intrinsics - result_native) < 0.00001) ? 
                                     "PASSED" : "FAILED");

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_r);
    free(a);
    free(b);
    free(r);

    return EXIT_SUCCESS;
}
