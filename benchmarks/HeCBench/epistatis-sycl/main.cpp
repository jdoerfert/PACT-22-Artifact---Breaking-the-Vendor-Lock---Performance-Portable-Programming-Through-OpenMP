
#include <math.h>
#include <float.h>
#include <fstream>
#include <string>
#include <cstring>
#include <sstream>
#include <iostream>
#include <chrono>
#include "common.h"

using namespace std::chrono;
typedef high_resolution_clock myclock;
typedef duration<float> myduration;

#define MAX_WG_SIZE 256

template <typename T>
T* mem_alloc (const int align, const size_t size) {
  return (T*) aligned_alloc(align, size * sizeof(T));
}

template <typename T>
void mem_free (T* p) {
  free(p);
}

float gammafunction(unsigned int n)
{   
  if(n == 0)
    return 0.0f;
  float x = ((float)n + 0.5f) * cl::sycl::log((float) n) - 
            ((float)n - 1.0f) * cl::sycl::log(cl::sycl::exp((float) 1.0f));
  return x;
}


int main(int argc, char **argv)
{
  int i, j, x;
  int num_pac = atoi(argv[1]);  // #samples
  int num_snp = atoi(argv[2]);  // #SNPs
  int iteration = atoi(argv[3]);// #kernel run
  int block_snp = 64;

  srand(100);
  unsigned char *SNP_Data = mem_alloc<unsigned char>(64, num_pac * num_snp);
  unsigned char *Ph_Data = mem_alloc<unsigned char>(64, num_pac);

  // generate SNPs between 0 and 2
  for (i = 0; i < num_pac; i++)
    for(j = 0; j < num_snp; j++)
      SNP_Data[i * num_snp + j] = rand() % 3;

  // generate phenotype between 0 and 1
  for(int i = 0; i < num_pac; i++) Ph_Data[i] = rand() % 2;

  // transpose the SNP data
  unsigned char *SNP_Data_trans = mem_alloc<unsigned char>(64, num_pac * num_snp);

  for (i = 0; i < num_pac; i++) 
    for(j = 0; j < num_snp; j++) 
      SNP_Data_trans[j * num_pac + i] = SNP_Data[i * num_snp + j];

  int phen_ones = 0;
  for(i = 0; i < num_pac; i++)
    if(Ph_Data[i] == 1)
      phen_ones++;

  // transform SNP data to a binary format

  int PP_zeros = ceil((1.0*(num_pac - phen_ones))/32.0);
  int PP_ones = ceil((1.0*phen_ones)/32.0);

  unsigned int *bin_data_zeros = mem_alloc<unsigned int>(64, num_snp * PP_zeros * 2);
  unsigned int *bin_data_ones = mem_alloc<unsigned int>(64, num_snp * PP_ones * 2);
  memset(bin_data_zeros, 0, num_snp*PP_zeros*2*sizeof(unsigned int));
  memset(bin_data_ones, 0, num_snp*PP_ones*2*sizeof(unsigned int));

  for(i = 0; i < num_snp; i++)
  {
    int x_zeros = -1;
    int x_ones = -1;
    int n_zeros = 0;
    int n_ones = 0;

    for(j = 0; j < num_pac; j++){
      unsigned int temp = (unsigned int) SNP_Data_trans[i * num_pac + j];

      if(Ph_Data[j] == 1){
        if(n_ones%32 == 0){
          x_ones ++;
        }
        // apply 1 shift left to 2 components
        bin_data_ones[i * PP_ones * 2 + x_ones*2 + 0] <<= 1;
        bin_data_ones[i * PP_ones * 2 + x_ones*2 + 1] <<= 1;
        // insert '1' in correct component
        if(temp == 0 || temp == 1){
          bin_data_ones[i * PP_ones * 2 + x_ones*2 + temp ] |= 1;
        }
        n_ones ++;
      } else {
        if(n_zeros%32 == 0){
          x_zeros ++;
        }
        // apply 1 shift left to 2 components
        bin_data_zeros[i * PP_zeros * 2 + x_zeros*2 + 0] <<= 1;
        bin_data_zeros[i * PP_zeros * 2 + x_zeros*2 + 1] <<= 1;
        // insert '1' in correct component
        if(temp == 0 || temp == 1){
          bin_data_zeros[i * PP_zeros * 2 + x_zeros*2 + temp] |= 1;
        }
        n_zeros ++;
      }
    }
  }

  unsigned int mask_zeros = 0xFFFFFFFF;
  for(int x = num_pac - phen_ones; x < PP_zeros * 32; x++)
    mask_zeros = mask_zeros >> 1;

  unsigned int mask_ones = 0xFFFFFFFF;
  for(x = phen_ones; x < PP_ones * 32; x++)
    mask_ones = mask_ones >> 1;

  // transpose the binary data structures
  unsigned int* bin_data_ones_trans = mem_alloc<unsigned int>(64, num_snp * PP_ones * 2);

  for(i = 0; i < num_snp; i++)
    for(j = 0; j < PP_ones; j++)
    {
      bin_data_ones_trans[(j * num_snp + i) * 2 + 0] = bin_data_ones[(i * PP_ones + j) * 2 + 0];
      bin_data_ones_trans[(j * num_snp + i) * 2 + 1] = bin_data_ones[(i * PP_ones + j) * 2 + 1];
    }

  unsigned int* bin_data_zeros_trans = mem_alloc<unsigned int>(64, num_snp * PP_zeros * 2);

  for(i = 0; i < num_snp; i++)
    for(j = 0; j < PP_zeros; j++)
    {
      bin_data_zeros_trans[(j * num_snp + i) * 2 + 0] = bin_data_zeros[(i * PP_zeros + j) * 2 + 0];
      bin_data_zeros_trans[(j * num_snp + i) * 2 + 1] = bin_data_zeros[(i * PP_zeros + j) * 2 + 1];
    }

  float *scores = mem_alloc<float>(64, num_snp * num_snp);
  for(x = 0; x < num_snp * num_snp; x++) scores[x] = FLT_MAX;

#ifdef USE_GPU
  gpu_selector dev_sel;
#else
  cpu_selector dev_sel;
#endif
  queue q(dev_sel);

  auto start = myclock::now();

  buffer<unsigned int, 1> d_data_zeros(bin_data_zeros_trans, num_snp * PP_zeros * 2);
  buffer<unsigned int, 1> d_data_ones(bin_data_ones_trans, num_snp * PP_ones * 2);
  buffer<float, 1> d_scores(num_snp * num_snp);

  // setup kernel ND-range
  int num_snp_m = num_snp;
  while(num_snp_m % block_snp != 0) num_snp_m++;
  range<2> global_epi(num_snp_m, num_snp_m);
  range<2> local_epi(1, block_snp);

  // epistasis detection kernel
  for (int i = 0; i < iteration; i++) {
  
    q.submit([&](handler& h) {
      auto acc = d_scores.get_access<sycl_discard_write>(h);
      h.copy(scores, acc);
    });

    q.submit([&](handler& h) {
      auto dev_data_zeros = d_data_zeros.get_access<sycl_read>(h);
      auto dev_data_ones = d_data_ones.get_access<sycl_read>(h);
      auto dev_scores = d_scores.get_access<sycl_write>(h);
      h.parallel_for<class kernel_epi>(nd_range<2>(global_epi, local_epi), [=](nd_item<2> id) {
        int i, j, tid, p, k;
        float score = FLT_MAX;

        i = id.get_global_id(0);
        j = id.get_global_id(1);
        tid = i * num_snp + j;

        if (j > i && i < num_snp && j < num_snp) {
          unsigned int ft[2 * 9];
          for(k = 0; k < 2 * 9; k++) ft[k] = 0;

          unsigned int t00, t01, t02, t10, t11, t12, t20, t21, t22;
          unsigned int di2, dj2;
          unsigned int* SNPi;
          unsigned int* SNPj;

          // Phenotype 0
          SNPi = (unsigned int*) &dev_data_zeros[i * 2];
          SNPj = (unsigned int*) &dev_data_zeros[j * 2];
          for (p = 0; p < 2 * PP_zeros * num_snp - 2 * num_snp; p += 2 * num_snp) {
            di2 = ~(SNPi[p] | SNPi[p + 1]);
            dj2 = ~(SNPj[p] | SNPj[p + 1]);

            t00 = SNPi[p] & SNPj[p];
            t01 = SNPi[p] & SNPj[p + 1];
            t02 = SNPi[p] & dj2;
            t10 = SNPi[p + 1] & SNPj[p];
            t11 = SNPi[p + 1] & SNPj[p + 1];
            t12 = SNPi[p + 1] & dj2;
            t20 = di2 & SNPj[p];
            t21 = di2 & SNPj[p + 1];
            t22 = di2 & dj2;

            ft[0] += cl::sycl::popcount(t00);
            ft[1] += cl::sycl::popcount(t01);
            ft[2] += cl::sycl::popcount(t02);
            ft[3] += cl::sycl::popcount(t10);
            ft[4] += cl::sycl::popcount(t11);
            ft[5] += cl::sycl::popcount(t12);
            ft[6] += cl::sycl::popcount(t20);
            ft[7] += cl::sycl::popcount(t21);
            ft[8] += cl::sycl::popcount(t22);
          }

          // remainder
          p = 2 * PP_zeros * num_snp - 2 * num_snp;
          di2 = ~(SNPi[p] | SNPi[p + 1]);
          dj2 = ~(SNPj[p] | SNPj[p + 1]);
          di2 = di2 & mask_zeros;
          dj2 = dj2 & mask_zeros;

          t00 = SNPi[p] & SNPj[p];
          t01 = SNPi[p] & SNPj[p + 1];
          t02 = SNPi[p] & dj2;
          t10 = SNPi[p + 1] & SNPj[p];
          t11 = SNPi[p + 1] & SNPj[p + 1];
          t12 = SNPi[p + 1] & dj2;
          t20 = di2 & SNPj[p];
          t21 = di2 & SNPj[p + 1];
          t22 = di2 & dj2;

          ft[0] += cl::sycl::popcount(t00);
          ft[1] += cl::sycl::popcount(t01);
          ft[2] += cl::sycl::popcount(t02);
          ft[3] += cl::sycl::popcount(t10);
          ft[4] += cl::sycl::popcount(t11);
          ft[5] += cl::sycl::popcount(t12);
          ft[6] += cl::sycl::popcount(t20);
          ft[7] += cl::sycl::popcount(t21);
          ft[8] += cl::sycl::popcount(t22);

          // Phenotype 1
          SNPi = (unsigned int*) &dev_data_ones[i * 2];
          SNPj = (unsigned int*) &dev_data_ones[j * 2];
          for(p = 0; p < 2 * PP_ones * num_snp - 2 * num_snp; p += 2 * num_snp)
          {
            di2 = ~(SNPi[p] | SNPi[p + 1]);
            dj2 = ~(SNPj[p] | SNPj[p + 1]);

            t00 = SNPi[p] & SNPj[p];
            t01 = SNPi[p] & SNPj[p + 1];
            t02 = SNPi[p] & dj2;
            t10 = SNPi[p + 1] & SNPj[p];
            t11 = SNPi[p + 1] & SNPj[p + 1];
            t12 = SNPi[p + 1] & dj2;
            t20 = di2 & SNPj[p];
            t21 = di2 & SNPj[p + 1];
            t22 = di2 & dj2;

            ft[9]  += cl::sycl::popcount(t00);
            ft[10] += cl::sycl::popcount(t01);
            ft[11] += cl::sycl::popcount(t02);
            ft[12] += cl::sycl::popcount(t10);
            ft[13] += cl::sycl::popcount(t11);
            ft[14] += cl::sycl::popcount(t12);
            ft[15] += cl::sycl::popcount(t20);
            ft[16] += cl::sycl::popcount(t21);
            ft[17] += cl::sycl::popcount(t22);
          }
          p = 2 * PP_ones * num_snp - 2 * num_snp;
          di2 = ~(SNPi[p] | SNPi[p + 1]);
          dj2 = ~(SNPj[p] | SNPj[p + 1]);
          di2 = di2 & mask_ones;
          dj2 = dj2 & mask_ones;

          t00 = SNPi[p] & SNPj[p];
          t01 = SNPi[p] & SNPj[p + 1];
          t02 = SNPi[p] & dj2;
          t10 = SNPi[p + 1] & SNPj[p];
          t11 = SNPi[p + 1] & SNPj[p + 1];
          t12 = SNPi[p + 1] & dj2;
          t20 = di2 & SNPj[p];
          t21 = di2 & SNPj[p + 1];
          t22 = di2 & dj2;

          ft[9]  += cl::sycl::popcount(t00);
          ft[10] += cl::sycl::popcount(t01);
          ft[11] += cl::sycl::popcount(t02);
          ft[12] += cl::sycl::popcount(t10);
          ft[13] += cl::sycl::popcount(t11);
          ft[14] += cl::sycl::popcount(t12);
          ft[15] += cl::sycl::popcount(t20);
          ft[16] += cl::sycl::popcount(t21);
          ft[17] += cl::sycl::popcount(t22);

          // compute score
          score = 0.0f;
          for(k = 0; k < 9; k++)
            score += gammafunction(ft[k] + ft[9 + k] + 1) - gammafunction(ft[k]) - gammafunction(ft[9 + k]);
          score = cl::sycl::fabs((float) score);
          if(score == 0.0f)
            score = FLT_MAX;
          dev_scores[tid] = score;
        }
      });
    });
  }

  q.submit([&](handler& h) {
    auto acc = d_scores.get_access<sycl_read>(h);
    h.copy(acc, scores);
  }).wait();

  auto end = myclock::now();
  myduration elapsed = end - start;
  std::cout << "Total offloading time: " << elapsed.count() << " sec" << std::endl;

  // compute the minimum score on a host
  float score = scores[0];
  int solution = 0;
  for (int i = 1; i < num_snp * num_snp; i++) {
    if (score > scores[i]) {
      score = scores[i];
      solution = i;
    }
  }

  std::cout << "Score: " << score << std::endl;
  std::cout << "Solution: " << solution / num_snp << ", " << solution % num_snp << std::endl;
  if ( (fabsf(score - 83.844f) > 1e-3f) || (solution / num_snp != 1253) || 
       (solution % num_snp != 25752) )
    printf("FAIL\n");
  else
    printf("PASS\n");

  mem_free(bin_data_zeros);
  mem_free(bin_data_ones);
  mem_free(bin_data_zeros_trans);
  mem_free(bin_data_ones_trans);
  mem_free(scores);
  mem_free(SNP_Data);
  mem_free(SNP_Data_trans);
  mem_free(Ph_Data);
  return 0;
}
