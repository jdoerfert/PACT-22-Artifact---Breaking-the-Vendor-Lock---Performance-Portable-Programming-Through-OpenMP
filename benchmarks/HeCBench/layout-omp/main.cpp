/**********************************************************************
* Copyright ©2013 Advanced Micro Devices, Inc. All rights reserved.
*
* Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
*
* Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
* Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or
*  other materials provided with the distribution.
*
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
* WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY
* DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
* OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
* NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
* ********************************************************************/


#include <iostream>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

#define TREE_NUM 4096
#define TREE_SIZE 4096
#define GROUP_SIZE 256

struct AppleTree
{
  int apples[TREE_SIZE];
};

struct ApplesOnTrees
{
  int trees[TREE_NUM];
};


int main(int argc, char * argv[])
{
  const int iterations = atoi(argv[1]); // Number of iterations for kernel execution
  const int treeSize = TREE_SIZE;
  const int treeNumber = TREE_NUM;
  bool fail = false;

  if(iterations < 1)
  {
    std::cout<<"Error, iterations cannot be 0 or negative. Exiting..\n";
    return -1;
  }

  if(treeNumber < GROUP_SIZE)
  {
    std::cout<<"Because of the work group size ,this treeNumber is small"<<std::endl;
    return -1;
  }
  if(treeNumber % 256 !=0)
  {
    std::cout<<"Because of the work group size ,should be a multiple of 256"<<std::endl;
    return -1;
  }

  const int elements = treeSize * treeNumber;
  size_t inputSize = elements * sizeof(int);
  size_t outputSize = treeNumber * sizeof(int);

  // input data
  int* data = (int*) malloc (inputSize);

  // device result
  int *output = (int *)malloc(outputSize);

  // compute reference for verification
  int *reference = (int *)malloc(outputSize);
  memset(reference,0,outputSize);
  for(int i=0; i < treeNumber; i++)
    for(int j=0; j < treeSize; j++)
      reference[i] += i * treeSize + j;

#pragma omp target data map(alloc: data[0:elements], output[0:treeNumber])
{
  //initialize aos data
  for (int i = 0; i < treeNumber; i++)
    for(int j = 0; j < treeSize; j++)
      data[j + i* treeSize] = j + i* treeSize;

  #pragma omp target update to (data[0:elements])

  AppleTree *trees = (AppleTree*) data;

  for (int n = 0; n < iterations; n++) {
    #pragma omp target teams distribute parallel for thread_limit(GROUP_SIZE) 
    for (uint gid = 0; gid < treeNumber; gid++) 
    {
      uint res = 0;
      for(int i = 0; i < treeSize; i++)
      {
        res += trees[gid].apples[i];
      }
      output[gid] = res;
    }
  }

  #pragma omp target update from (output[0:treeNumber])

  for(int i=0; i< treeNumber; i++)
  {
    if(output[i] != reference[i])
    {
      fail = true;
      break;
    }
  }

  if (fail)
    std::cout << "Failed\n";
  else
    std::cout << "Passed\n";

  //initialize soa data
  for (int i = 0; i < treeNumber; i++)
    for(int j = 0; j < treeSize; j++)
      data[i + j* treeNumber] = j + i* treeSize;

  #pragma omp target update to (data[0:elements])

  ApplesOnTrees *applesOnTrees = (ApplesOnTrees*) data;

  for (int n = 0; n < iterations; n++) {
    #pragma omp target teams distribute parallel for thread_limit(GROUP_SIZE) 
    for (uint gid = 0; gid < treeNumber; gid++) 
    {
      uint res = 0;
      for(int i = 0; i < treeSize; i++)
      {
        res += applesOnTrees[i].trees[gid];
      }
      output[gid] = res;
    }
  }

  #pragma omp target update from (output[0:treeNumber])

  for(int i=0; i< treeNumber; i++)
  {
    if(output[i] != reference[i])
    {
      fail = true;
      break;
    }
  }

  if (fail)
    std::cout << "Failed\n";
  else
    std::cout << "Passed\n";

}

  free(output);
  free(reference);
  free(data);
  return 0;
}

