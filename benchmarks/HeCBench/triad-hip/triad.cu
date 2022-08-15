#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <hip/hip_runtime.h>
#include <chrono>

#include "OptionParser.h"
#include "Timer.h"
#include "Utility.h"

// ****************************************************************************
// Function: addBenchmarkSpecOptions
//
// Purpose:
//   Add benchmark specific options parsing
//
// Arguments:
//   op: the options parser / parameter database
//
// Returns:  nothing
//
// Programmer: Kyle Spafford
// Creation: December 15, 2009
//
// Modifications:
//
// ****************************************************************************
void addBenchmarkSpecOptions(OptionParser &op)
{
  ;
}

// ****************************************************************************
// Function: triad
//
// Purpose:
//   A simple vector addition kernel
//   C = A + s*B
//
// Arguments:
//   A,B - input vectors
//   C - output vectors
//   s - scalar
//
// Returns:  nothing
//
// Programmer: Kyle Spafford
// Creation: December 15, 2009
//
// Modifications:
//
// ****************************************************************************
__global__ void triad(float* A, float* B, float* C, float s)
{
  int gid = threadIdx.x + (blockIdx.x * blockDim.x);
  C[gid] = A[gid] + s*B[gid];
}

// ****************************************************************************
// Function: RunBenchmark
//
// Purpose:
//   Implements the Stream Triad benchmark in CUDA.  This benchmark
//   is designed to test CUDA's overall data transfer speed. It executes
//   a vector addition operation with no temporal reuse. Data is read
//   directly from the global memory. This implementation tiles the input
//   array and pipelines the vector addition computation with
//   the data download for the next tile. However, since data transfer from
//   host to device is much more expensive than the simple vector computation,
//   data transfer operations should completely dominate the execution time.
//
// Arguments:
//   resultDB: results from the benchmark are stored in this db
//   op: the options parser (contains input parameters)
//
// Returns:  nothing
//
// Programmer: Kyle Spafford
// Creation: December 15, 2009
//
// Modifications:
//
// ****************************************************************************
void RunBenchmark(OptionParser &op)
{
  auto start = std::chrono::high_resolution_clock::now();

  const bool verbose = op.getOptionBool("verbose");
  const int n_passes = op.getOptionInt("passes");

  // 256k through 8M bytes
  const int nSizes = 9;
  const size_t blockSizes[] = { 64, 128, 256, 512, 1024, 2048, 4096, 8192,
    16384 };
  const size_t memSize = 16384;
  const size_t numMaxFloats = 1024 * memSize / 4;
  const size_t halfNumFloats = numMaxFloats / 2;

  // Create some host memory pattern
  srand48(8650341L);
  float *h_mem;
  hipHostMalloc((void**)&h_mem, sizeof(float) * numMaxFloats);

  // Allocate some device memory
  float* d_memA0, *d_memB0, *d_memC0;
  hipMalloc((void**) &d_memA0, blockSizes[nSizes - 1] * 1024);
  hipMalloc((void**) &d_memB0, blockSizes[nSizes - 1] * 1024);
  hipMalloc((void**) &d_memC0, blockSizes[nSizes - 1] * 1024);

  float* d_memA1, *d_memB1, *d_memC1;
  hipMalloc((void**) &d_memA1, blockSizes[nSizes - 1] * 1024);
  hipMalloc((void**) &d_memB1, blockSizes[nSizes - 1] * 1024);
  hipMalloc((void**) &d_memC1, blockSizes[nSizes - 1] * 1024);

  float scalar = 1.75f;

  const size_t blockSize = 128;

  // Number of passes. Use a large number for stress testing.
  // A small value is sufficient for computing sustained performance.
  for (int pass = 0; pass < n_passes; ++pass)
  {
    // Step through sizes forward
    for (int i = 0; i < nSizes; ++i)
    {
      int elemsInBlock = blockSizes[i] * 1024 / sizeof(float);
      for (int j = 0; j < halfNumFloats; ++j)
        h_mem[j] = h_mem[halfNumFloats + j]
          = (float) (drand48() * 10.0);

      // Copy input memory to the device
      if (verbose) {
        cout << ">> Executing Triad with vectors of length "
          << numMaxFloats << " and block size of "
          << elemsInBlock << " elements." << "\n";
        printf("Block:%05ldKB\n", blockSizes[i]);
      }

      // start submitting blocks of data of size elemsInBlock
      // overlap the computation of one block with the data
      // download for the next block and the results upload for
      // the previous block
      int crtIdx = 0;
      size_t globalWorkSize = elemsInBlock / blockSize;

      hipStream_t streams[2];
      hipStreamCreate(&streams[0]);
      hipStreamCreate(&streams[1]);

      int TH = Timer::Start();

      hipMemcpyAsync(d_memA0, h_mem, blockSizes[i] * 1024,
          hipMemcpyHostToDevice, streams[0]);
      hipMemcpyAsync(d_memB0, h_mem, blockSizes[i] * 1024,
          hipMemcpyHostToDevice, streams[0]);

      hipLaunchKernelGGL(triad, dim3(globalWorkSize), dim3(blockSize), 0, streams[0], d_memA0, d_memB0, d_memC0, scalar);

      if (elemsInBlock < numMaxFloats)
      {
        // start downloading data for next block
        hipMemcpyAsync(d_memA1, h_mem + elemsInBlock, blockSizes[i]
            * 1024, hipMemcpyHostToDevice, streams[1]);
        hipMemcpyAsync(d_memB1, h_mem + elemsInBlock, blockSizes[i]
            * 1024, hipMemcpyHostToDevice, streams[1]);
      }

      int blockIdx = 1;
      unsigned int currStream = 1;
      while (crtIdx < numMaxFloats)
      {
        currStream = blockIdx & 1;
        // Start copying back the answer from the last kernel
        if (currStream)
        {
          hipMemcpyAsync(h_mem + crtIdx, d_memC0, elemsInBlock
              * sizeof(float), hipMemcpyDeviceToHost, streams[0]);
        }
        else
        {
          hipMemcpyAsync(h_mem + crtIdx, d_memC1, elemsInBlock
              * sizeof(float), hipMemcpyDeviceToHost, streams[1]);
        }

        crtIdx += elemsInBlock;

        if (crtIdx < numMaxFloats)
        {
          // Execute the kernel
          if (currStream)
          {
            hipLaunchKernelGGL(triad, dim3(globalWorkSize), dim3(blockSize), 0, streams[1], d_memA1, d_memB1, d_memC1, scalar);
          }
          else
          {
            hipLaunchKernelGGL(triad, dim3(globalWorkSize), dim3(blockSize), 0, streams[0], d_memA0, d_memB0, d_memC0, scalar);
          }
        }

        if (crtIdx+elemsInBlock < numMaxFloats)
        {
          // Download data for next block
          if (currStream)
          {
            hipMemcpyAsync(d_memA0, h_mem+crtIdx+elemsInBlock,
                blockSizes[i]*1024, hipMemcpyHostToDevice,
                streams[0]);
            hipMemcpyAsync(d_memB0, h_mem+crtIdx+elemsInBlock,
                blockSizes[i]*1024, hipMemcpyHostToDevice,
                streams[0]);
          }
          else
          {
            hipMemcpyAsync(d_memA1, h_mem+crtIdx+elemsInBlock,
                blockSizes[i]*1024, hipMemcpyHostToDevice,
                streams[1]);
            hipMemcpyAsync(d_memB1, h_mem+crtIdx+elemsInBlock,
                blockSizes[i]*1024, hipMemcpyHostToDevice,
                streams[1]);
          }
        }
        blockIdx += 1;
        currStream = !currStream;
      }

      hipDeviceSynchronize();
      double time = Timer::Stop(TH, "thread synchronize");

      double triad = ((double)numMaxFloats * 2.0) / (time*1e9);
      if (verbose)
        std::cout << "TriadFlops " << triad << " GFLOPS/s\n";

      double bdwth = ((double)numMaxFloats*sizeof(float)*3.0)
        / (time*1000.*1000.*1000.);
      if (verbose)
        std::cout << "TriadBdwth " << bdwth << " GB/s\n";


      // Checking memory for correctness. The two halves of the array
      // should have the same results.
      if (verbose) cout << ">> checking memory\n";
      for (int j=0; j<halfNumFloats; ++j)
      {
        if (h_mem[j] != h_mem[j+halfNumFloats])
        {
          cout << "Error; hostMem[" << j << "]=" << h_mem[j]
            << " is different from its twin element hostMem["
            << (j+halfNumFloats) << "]: "
            << h_mem[j+halfNumFloats] << "stopping check\n";
          break;
        }
      }
      if (verbose) cout << ">> finish!" << endl;

      // Zero out the test host memory
      for (int j=0; j<numMaxFloats; ++j)
        h_mem[j] = 0.0f;
    }
  }

  // Cleanup
  hipFree(d_memA0);
  hipFree(d_memB0);
  hipFree(d_memC0);
  hipFree(d_memA1);
  hipFree(d_memB1);
  hipFree(d_memC1);
  hipHostFree(h_mem);

  auto end = std::chrono::high_resolution_clock::now();
  auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
  printf("Total execution time (function 'RunBechmark') (in ms): %ld \n", elapsed);
}
