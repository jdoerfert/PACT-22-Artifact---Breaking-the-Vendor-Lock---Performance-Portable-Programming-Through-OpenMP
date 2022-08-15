#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <time.h>
#include <hip/hip_runtime.h>
#include "benchmark.h"
#include "kernels.h"

void run_benchmark()
{
  int i, j, cnt, val_ref, val_eff;
  uint64_t time_vals[SIZES_CNT_MAX][BASES_CNT_MAX][2];

  uint32_t *d_bases32;
  uint32_t bases32_size = sizeof(bases32);
  hipMalloc((void**)&d_bases32, bases32_size);
  hipMemcpy(d_bases32, bases32, bases32_size, hipMemcpyHostToDevice);

  uint32_t *d_n32;
  uint32_t n32_size = BENCHMARK_ITERATIONS * sizeof(uint32_t);
  hipMalloc((void**)&d_n32, n32_size);

  int *d_val;
  int val_dev;
  hipMalloc((void**)&d_val, sizeof(int));
  dim3 grids ((BENCHMARK_ITERATIONS + 255) / 256);
  dim3 blocks (256);

  printf("Starting benchmark...\n");

  for (i = 0; i < SIZES_CNT32; i++) {
    val_ref = val_eff = 0;
    hipMemcpy(d_n32, n32[i], n32_size, hipMemcpyHostToDevice);
    hipMemset(d_val, 0, sizeof(int));

    for (cnt = 1; cnt <= BASES_CNT32; cnt++) {
      time_point start = get_time();
      for (j = 0; j < BENCHMARK_ITERATIONS; j++)
        val_eff += efficient_mr32(bases32, cnt, n32[i][j]);
      time_vals[i][cnt - 1][0] = elapsed_time(start);
    }

    for (cnt = 1; cnt <= BASES_CNT32; cnt++) {
      time_point start = get_time();
      for (j = 0; j < BENCHMARK_ITERATIONS; j++)
        val_ref += straightforward_mr32(bases32, cnt, n32[i][j]);
      time_vals[i][cnt - 1][1] = elapsed_time(start);
    }

    // verify the results of simple and efficient versions on a host
    if (val_ref != val_eff) {
      fprintf(stderr, "Results mismatch: val_ref = %d, val_eff = %d\n", val_ref, val_eff);
      break;
    }

    // the efficient version is faster than the simple version on a device
    hipLaunchKernelGGL(mr32_sf, dim3(grids), dim3(blocks ), 0, 0, d_bases32, d_n32, d_val, BENCHMARK_ITERATIONS);
    hipMemcpy(&val_dev, d_val, sizeof(int), hipMemcpyDeviceToHost);
    if (val_ref != val_dev) {
      fprintf(stderr, "Results mismatch: val_dev = %d, val_ref = %d\n", val_dev, val_ref);
      break;
    }

    hipMemset(d_val, 0, sizeof(int));
    hipLaunchKernelGGL(mr32_eff, dim3(grids), dim3(blocks ), 0, 0, d_bases32, d_n32, d_val, BENCHMARK_ITERATIONS);
    hipMemcpy(&val_dev, d_val, sizeof(int), hipMemcpyDeviceToHost);
    if (val_ref != val_dev) {
      fprintf(stderr, "Results mismatch: val_dev = %d, val_ref = %d\n", val_dev, val_ref);
      break;
    }
  }

  // device results are not included
  print_results(bits32, SIZES_CNT32, BASES_CNT32, time_vals);

  hipFree(d_bases32);
  hipFree(d_n32);
  hipFree(d_val);
}

int main()
{
#ifdef _WIN32
  system("mode CON: COLS=98");
#endif

  printf("Setting random primes...\n");
  set_nprimes();
  run_benchmark();

  printf("Setting random odd integers...\n");
  set_nintegers();
  run_benchmark();

  return 0;
}
