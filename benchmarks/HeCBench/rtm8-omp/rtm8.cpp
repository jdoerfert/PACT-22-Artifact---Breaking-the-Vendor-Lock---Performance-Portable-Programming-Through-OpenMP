#include <iostream>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

#define nt 30
#define nx 680
#define ny 134
#define nz 450

#include "mysecond.c"

inline int indexTo1D(int x, int y, int z){
  return x + y*nx + z*nx*ny;
}

void rtm8_cpu(float* vsq, float* current_s, float* current_r, float* next_s, float* next_r, float* image, float* a, size_t N)
{
#ifdef _OPENMP
#pragma omp parallel for collapse(3)
#endif
  for (int z = 4; z < nz - 4; z++) {
    for (int y = 4; y < ny - 4; y++) {
      for (int x = 4; x < nx - 4; x++) {
        float div =
          a[0] * current_s[indexTo1D(x,y,z)] +
          a[1] * (current_s[indexTo1D(x+1,y,z)] + current_s[indexTo1D(x-1,y,z)] +
              current_s[indexTo1D(x,y+1,z)] + current_s[indexTo1D(x,y-1,z)] +
              current_s[indexTo1D(x,y,z+1)] + current_s[indexTo1D(x,y,z-1)]) +
          a[2] * (current_s[indexTo1D(x+2,y,z)] + current_s[indexTo1D(x-2,y,z)] +
              current_s[indexTo1D(x,y+2,z)] + current_s[indexTo1D(x,y-2,z)] +
              current_s[indexTo1D(x,y,z+2)] + current_s[indexTo1D(x,y,z-2)]) +
          a[3] * (current_s[indexTo1D(x+3,y,z)] + current_s[indexTo1D(x-3,y,z)] +
              current_s[indexTo1D(x,y+3,z)] + current_s[indexTo1D(x,y-3,z)] +
              current_s[indexTo1D(x,y,z+3)] + current_s[indexTo1D(x,y,z-3)]) +
          a[4] * (current_s[indexTo1D(x+4,y,z)] + current_s[indexTo1D(x-4,y,z)] +
              current_s[indexTo1D(x,y+4,z)] + current_s[indexTo1D(x,y-4,z)] +
              current_s[indexTo1D(x,y,z+4)] + current_s[indexTo1D(x,y,z-4)]);

        next_s[indexTo1D(x,y,z)] = 2*current_s[indexTo1D(x,y,z)] - next_s[indexTo1D(x,y,z)]
          + vsq[indexTo1D(x,y,z)]*div;
        div =
          a[0] * current_r[indexTo1D(x,y,z)] +
          a[1] * (current_r[indexTo1D(x+1,y,z)] + current_r[indexTo1D(x-1,y,z)] +
              current_r[indexTo1D(x,y+1,z)] + current_r[indexTo1D(x,y-1,z)] +
              current_r[indexTo1D(x,y,z+1)] + current_r[indexTo1D(x,y,z-1)]) +
          a[2] * (current_r[indexTo1D(x+2,y,z)] + current_r[indexTo1D(x-2,y,z)] +
              current_r[indexTo1D(x,y+2,z)] + current_r[indexTo1D(x,y-2,z)] +
              current_r[indexTo1D(x,y,z+2)] + current_r[indexTo1D(x,y,z-2)]) +
          a[3] * (current_r[indexTo1D(x+3,y,z)] + current_r[indexTo1D(x-3,y,z)] +
              current_r[indexTo1D(x,y+3,z)] + current_r[indexTo1D(x,y-3,z)] +
              current_r[indexTo1D(x,y,z+3)] + current_r[indexTo1D(x,y,z-3)]) +
          a[4] * (current_r[indexTo1D(x+4,y,z)] + current_r[indexTo1D(x-4,y,z)] +
              current_r[indexTo1D(x,y+4,z)] + current_r[indexTo1D(x,y-4,z)] +
              current_r[indexTo1D(x,y,z+4)] + current_r[indexTo1D(x,y,z-4)]);

        next_r[indexTo1D(x,y,z)] = 2 * current_r[indexTo1D(x,y,z)]
          - next_r[indexTo1D(x,y,z)] + vsq[indexTo1D(x,y,z)] * div;

        image[indexTo1D(x,y,z)] = next_s[indexTo1D(x,y,z)] * next_r[indexTo1D(x,y,z)];
      }
    }
  }
}


int main() {
  const int ArraySize = nx * ny * nz;
  float* next_s = (float*)malloc(ArraySize * sizeof(float));
  float* current_s = (float*)malloc(ArraySize * sizeof(float));
  float* next_r = (float*)malloc(ArraySize * sizeof(float));
  float* current_r = (float*)malloc(ArraySize * sizeof(float));
  float* vsq = (float*)malloc(ArraySize * sizeof(float));
  float* image_gpu = (float*)malloc(ArraySize * sizeof(float));
  float* image_cpu = (float*)malloc(ArraySize * sizeof(float));

  float a[5];
  double pts, t0, t1, dt, flops, pt_rate, flop_rate, speedup, memory;

  memory = ArraySize*sizeof(float)*6; 
  pts = nt;
  pts = pts*(nx-8)*(ny-8)*(nz-8);
  flops = 67*pts;
  printf("memory (MB) = %f\n", memory/1e6);
  printf("pts (billions) = %f\n", pts/1e9);
  printf("Tflops = %f\n", flops/1e12);

  // Initialization of matrix
  a[0] = -1./560.;
  a[1] = 8./315;
  a[2] = -0.2;
  a[3] = 1.6;
  a[4] = -1435./504.;

  for (int z = 0; z < nz; z++) {
    for (int y = 0; y < ny; y++) {
      for (int x = 0; x < nx; x++) {
        vsq[indexTo1D(x,y,z)] = 1.0;
        next_s[indexTo1D(x,y,z)] = 0;
        current_s[indexTo1D(x,y,z)] = 1.0;
        next_r[indexTo1D(x,y,z)] = 0;
        current_r[indexTo1D(x,y,z)] = 1.0;
        image_gpu[indexTo1D(x,y,z)] = image_cpu[indexTo1D(x,y,z)] = 0.5;
      }
    }
  }

  t0 = mysecond();
#pragma omp target data map(to: current_s[0:ArraySize]) \
                        map(to: current_r[0:ArraySize]) \
                        map(to: a[0:5]) \
                        map(to: vsq[0:ArraySize]) \
                        map(alloc: next_r[0:ArraySize]) \
                        map(alloc: next_s[0:ArraySize]) \
                        map(tofrom: image_gpu[0:ArraySize]) 
{
  for (int t = 0; t < nt; t++) {
    #pragma omp target teams distribute parallel for collapse(3) thread_limit(256)
    for (int z = 4; z < nz - 4; z++) {
      for (int y = 4; y < ny - 4; y++) {
        for (int x = 4; x < nx - 4; x++) {
          float div =
            a[0] * current_s[indexTo1D(x,y,z)] +
            a[1] * (current_s[indexTo1D(x+1,y,z)] + current_s[indexTo1D(x-1,y,z)] +
                current_s[indexTo1D(x,y+1,z)] + current_s[indexTo1D(x,y-1,z)] +
                current_s[indexTo1D(x,y,z+1)] + current_s[indexTo1D(x,y,z-1)]) +
            a[2] * (current_s[indexTo1D(x+2,y,z)] + current_s[indexTo1D(x-2,y,z)] +
                current_s[indexTo1D(x,y+2,z)] + current_s[indexTo1D(x,y-2,z)] +
                current_s[indexTo1D(x,y,z+2)] + current_s[indexTo1D(x,y,z-2)]) +
            a[3] * (current_s[indexTo1D(x+3,y,z)] + current_s[indexTo1D(x-3,y,z)] +
                current_s[indexTo1D(x,y+3,z)] + current_s[indexTo1D(x,y-3,z)] +
                current_s[indexTo1D(x,y,z+3)] + current_s[indexTo1D(x,y,z-3)]) +
            a[4] * (current_s[indexTo1D(x+4,y,z)] + current_s[indexTo1D(x-4,y,z)] +
                current_s[indexTo1D(x,y+4,z)] + current_s[indexTo1D(x,y-4,z)] +
                current_s[indexTo1D(x,y,z+4)] + current_s[indexTo1D(x,y,z-4)]);

          next_s[indexTo1D(x,y,z)] = 2*current_s[indexTo1D(x,y,z)] - next_s[indexTo1D(x,y,z)]
            + vsq[indexTo1D(x,y,z)]*div;
          div =
            a[0] * current_r[indexTo1D(x,y,z)] +
            a[1] * (current_r[indexTo1D(x+1,y,z)] + current_r[indexTo1D(x-1,y,z)] +
                current_r[indexTo1D(x,y+1,z)] + current_r[indexTo1D(x,y-1,z)] +
                current_r[indexTo1D(x,y,z+1)] + current_r[indexTo1D(x,y,z-1)]) +
            a[2] * (current_r[indexTo1D(x+2,y,z)] + current_r[indexTo1D(x-2,y,z)] +
                current_r[indexTo1D(x,y+2,z)] + current_r[indexTo1D(x,y-2,z)] +
                current_r[indexTo1D(x,y,z+2)] + current_r[indexTo1D(x,y,z-2)]) +
            a[3] * (current_r[indexTo1D(x+3,y,z)] + current_r[indexTo1D(x-3,y,z)] +
                current_r[indexTo1D(x,y+3,z)] + current_r[indexTo1D(x,y-3,z)] +
                current_r[indexTo1D(x,y,z+3)] + current_r[indexTo1D(x,y,z-3)]) +
            a[4] * (current_r[indexTo1D(x+4,y,z)] + current_r[indexTo1D(x-4,y,z)] +
                current_r[indexTo1D(x,y+4,z)] + current_r[indexTo1D(x,y-4,z)] +
                current_r[indexTo1D(x,y,z+4)] + current_r[indexTo1D(x,y,z-4)]);

          next_r[indexTo1D(x,y,z)] = 2 * current_r[indexTo1D(x,y,z)]
            - next_r[indexTo1D(x,y,z)] + vsq[indexTo1D(x,y,z)] * div;

          image_gpu[indexTo1D(x,y,z)] = next_s[indexTo1D(x,y,z)] * next_r[indexTo1D(x,y,z)];
	}
      }
    }
  }
  }
  t1 = mysecond();
  dt = t1 - t0;

  t0 = mysecond();
  for (int t = 0; t < nt; t++) {
    rtm8_cpu(vsq, current_s, next_s, current_r, next_r, image_cpu, a, ArraySize);
  }
  t1 = mysecond();

  // verification
  for (int i = 0; i < ArraySize; i++) 
    if (fabsf(image_cpu[i] - image_gpu[i]) > 0.1) {
      printf("@index %d cpu: %f gpu %f\n", i, image_cpu[i], image_gpu[i]);
      break;
    }

  pt_rate = pts/dt;
  flop_rate = flops/dt;
  printf("dt = %f\n", dt);
  printf("pt_rate (millions/sec) = %f\n", pt_rate/1e6);
  printf("flop_rate (Gflops) = %f\n", flop_rate/1e9);
  printf("speedup over cpu = %f\n", (t1 - t0) / dt);

  //release arrays
  free(vsq);
  free(next_s);
  free(current_s);
  free(next_r);
  free(current_r);
  free(image_cpu);
  free(image_gpu);
  return 0;

}

