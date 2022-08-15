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

///////////////////////////////////////////////////////////////////////////////
// This sample implements Mersenne Twister random number generator
// and Cartesian Box-Muller transformation on the GPU
///////////////////////////////////////////////////////////////////////////////

// standard utilities and systems includes
#include <stdio.h>
#include "MT.h"
#include "common.h"

// comment the below line if not doing Box-Muller transformation
#define DO_BOXMULLER

// Reference CPU MT and Box-Muller transformation 
extern "C" void initMTRef(const char *fname);
extern "C" void RandomRef(float *h_Rand, int nPerRng, unsigned int seed);
#ifdef DO_BOXMULLER
extern "C" void BoxMullerRef(float *h_Rand, int nPerRng);
#endif

#include <chrono>
using namespace std::chrono;

///////////////////////////////////////////////////////////////////////////////
//Load twister configurations
///////////////////////////////////////////////////////////////////////////////
void loadMTGPU(const char *fname, 
	       const unsigned int seed, 
	       mt_struct_stripped *h_MT,
	       const size_t size)
{
    // open the file for binary read
    FILE* fd = fopen(fname, "rb");
    if (fd == NULL) 
    {
      printf("Failed to open file %s\n", fname);
      exit(-1);
    }
  
    for (unsigned int i = 0; i < size; i++)
        fread(&h_MT[i], sizeof(mt_struct_stripped), 1, fd);
    fclose(fd);

    for(unsigned int i = 0; i < size; i++)
        h_MT[i].seed = seed;
}


void BoxMullerTrans(float *u1, float *u2)
{
    const float   r = cl::sycl::sqrt(-2.0f * cl::sycl::log(*u1));
    const float phi = 2 * PI * (*u2);
    *u1 = r * cl::sycl::cos(phi);
    *u2 = r * cl::sycl::sin(phi);
}

///////////////////////////////////////////////////////////////////////////////
// Main function 
///////////////////////////////////////////////////////////////////////////////
int main(int argc, const char **argv)
{
    size_t globalWorkSize = {MT_RNG_COUNT};  // 1D var for Total # of work items
    size_t localWorkSize = {128};            // 1D var for # of work items in the work group	
    const int seed = 777;
    const int nPerRng = 5860;                // # of recurrence steps, must be even if do Box-Muller transformation
    const int nRand = MT_RNG_COUNT * nPerRng;// Output size

    printf("Initialization: load MT parameters and init host buffers...\n");
    mt_struct_stripped *h_MT = (mt_struct_stripped*) malloc (
                               sizeof(mt_struct_stripped) * MT_RNG_COUNT); // MT para

    const char *cDatPath = "./data/MersenneTwister.dat";
    loadMTGPU(cDatPath, seed, h_MT, MT_RNG_COUNT);

    const char *cRawPath = "./data/MersenneTwister.raw";
    initMTRef(cRawPath);

    float *h_RandGPU = (float*)malloc(sizeof(float)*nRand); // Host buffers for GPU output
    float *h_RandCPU = (float*)malloc(sizeof(float)*nRand); // Host buffers for CPU test

    printf("Allocate memory...\n"); 
    
    buffer<mt_struct_stripped, 1> d_MT (h_MT, MT_RNG_COUNT); 
    buffer<float, 1> d_Rand (nRand);

#ifdef USE_GPU
    gpu_selector dev_sel;
#else
    cpu_selector dev_sel;
#endif
    queue q(dev_sel);

    int numIterations = 100;
    printf("Call Mersenne Twister kernel... (%d iterations)\n\n", numIterations); 
    high_resolution_clock::time_point t1 = high_resolution_clock::now();
    for (int i = -1; i < numIterations; i++)
    {
        q.submit([&] (handler &cgh) {
           auto Rand = d_Rand.get_access<sycl_discard_write>(cgh); 
           auto MT = d_MT.get_access<sycl_read>(cgh); 
           cgh.parallel_for<class mt>(nd_range<1>(range<1>(globalWorkSize), 
                                                  range<1>(localWorkSize)), [=] (nd_item<1> item) {
              int globalID = item.get_global_id(0);

              int iState, iState1, iStateM, iOut;
              unsigned int mti, mti1, mtiM, x;
              unsigned int mt[MT_NN], matrix_a, mask_b, mask_c; 

              //Load bit-vector Mersenne Twister parameters
              matrix_a = MT[globalID].matrix_a;
              mask_b   = MT[globalID].mask_b;
              mask_c   = MT[globalID].mask_c;
                  
              //Initialize current state
              mt[0] = MT[globalID].seed;
              for (iState = 1; iState < MT_NN; iState++)
                  mt[iState] = (1812433253U * (mt[iState - 1] ^ (mt[iState - 1] >> 30)) + iState) & MT_WMASK;

              iState = 0;
              mti1 = mt[0];
              for (iOut = 0; iOut < nPerRng; iOut++) {
                  iState1 = iState + 1;
                  iStateM = iState + MT_MM;
                  if(iState1 >= MT_NN) iState1 -= MT_NN;
                  if(iStateM >= MT_NN) iStateM -= MT_NN;
                  mti  = mti1;
                  mti1 = mt[iState1];
                  mtiM = mt[iStateM];

                      // MT recurrence
                  x = (mti & MT_UMASK) | (mti1 & MT_LMASK);
                      x = mtiM ^ (x >> 1) ^ ((x & 1) ? matrix_a : 0);

                  mt[iState] = x;
                  iState = iState1;

                  //Tempering transformation
                  x ^= (x >> MT_SHIFT0);
                  x ^= (x << MT_SHIFTB) & mask_b;
                  x ^= (x << MT_SHIFTC) & mask_c;
                  x ^= (x >> MT_SHIFT1);

                  //Convert to (0, 1] float and write to global memory
                  Rand[globalID + iOut * MT_RNG_COUNT] = ((float)x + 1.0f) / 4294967296.0f;
              }
           });
         });
    	
    #ifdef DO_BOXMULLER 
        q.submit([&] (handler &cgh) {
           auto Rand = d_Rand.get_access<sycl_read_write>(cgh); 
           cgh.parallel_for<class boxmuller>(nd_range<1>(range<1>(globalWorkSize), 
                                                  range<1>(localWorkSize)), [=] (nd_item<1> item) {
              int globalID = item.get_global_id(0);
              for (int iOut = 0; iOut < nPerRng; iOut += 2) {
                 BoxMullerTrans(&Rand[globalID + (iOut + 0) * MT_RNG_COUNT],
		       &Rand[globalID + (iOut + 1) * MT_RNG_COUNT]);
              }
           });
        });
    #endif
    }
    q.wait();
    
    high_resolution_clock::time_point t2 = high_resolution_clock::now();
    duration<double> time_span = duration_cast<duration<double>>(t2 - t1);
    double gpuTime = time_span.count() / (double)numIterations;
    printf("MersenneTwister, Throughput = %.4f GNumbers/s, "
           "Time = %.5f s, Size = %u Numbers, Workgroup = %lu\n", 
           ((double)nRand * 1.0E-9 / gpuTime), gpuTime, nRand, localWorkSize);    

    printf("\nRead back results...\n"); 
    q.submit([&] (handler &cgh) {
      auto acc = d_Rand.get_access<sycl_read>(cgh); 
      cgh.copy(acc, h_RandGPU);
    });
    q.wait();

    printf("Compute CPU reference solution...\n");
    {
        RandomRef(h_RandCPU, nPerRng, seed);
#ifdef DO_BOXMULLER
        BoxMullerRef(h_RandCPU, nPerRng);
#endif
    }

    printf("Compare CPU and GPU results...\n");
    double sum_delta = 0;
    double sum_ref   = 0;
    {
        for(int i = 0; i < MT_RNG_COUNT; i++)
            for(int j = 0; j < nPerRng; j++) {
	        double rCPU = h_RandCPU[i * nPerRng + j];
	        double rGPU = h_RandGPU[i + j * MT_RNG_COUNT];
	        double delta = std::fabs(rCPU - rGPU);
	        sum_delta += delta;
	        sum_ref   += std::fabs(rCPU);
	    }
    }
    double L1norm = sum_delta / sum_ref;
    printf("L1 norm: %E\n\n", L1norm);

    free(h_MT);
    free(h_RandGPU);
    free(h_RandCPU);

    // finish
    if (L1norm < 1e-6)
      printf("PASSED\n");
    else
      printf("FAILED\n");

    return 0;
}
