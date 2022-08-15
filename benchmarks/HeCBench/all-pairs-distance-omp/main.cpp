/* This code is provided as supplementary material for the book
   chapter "Exploiting graphics processing units for computational
   biology and bioinformatics," by Payne, Sinnott-Armstrong, and
   Moore, to appear in "The Handbook of Research on Computational and
   Systems Biology: Interdisciplinary applications," by IGI Global.

   Please feel free to use, modify, or redistribute this code.

   Make sure you have a CUDA compatible GPU and the nvcc is installed.
   To compile, type make.
   After compilation, type ./chapter to run
   Output written to timing.txt
*/

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <omp.h>
#include <sys/time.h>

#define INSTANCES 224   /* # of instances */
#define ATTRIBUTES 4096 /* # of attributes */
#define THREADS 128    /* # of threads per block */

struct char4 { char x; char y; char z; char w; };


/* CPU implementation */
void CPU(int * data, int * distance) {
  /* compare all pairs of instances, accessing the attributes in
     row-major order */
  #pragma omp parallel for collapse(2)
  for (int i = 0; i < INSTANCES; i++) {
    for (int j = 0; j < INSTANCES; j++) {
      for (int k = 0; k < ATTRIBUTES; k++) {
        distance[i + INSTANCES * j] += 
          (data[i * ATTRIBUTES + k] != data[j * ATTRIBUTES + k]);
      }
    }
  }
}

int main(int argc, char **argv) {

  /* host data */
  int *data; 
  char *data_char;
  int *cpu_distance, *gpu_distance;

  /* used to time CPU and GPU implementations */
  double start_cpu, stop_cpu;
  double start_gpu, stop_gpu;
  double elapsedTime; 
  struct timeval tp;
  struct timezone tzp;
  /* verification result */ 
  int status;
  /* output file for timing results */
  FILE *out = fopen("timing.txt","a");

  /* seed RNG */
  srand(2);

  /* allocate host memory */
  data = (int *)malloc(INSTANCES * ATTRIBUTES * sizeof(int));
  data_char = (char *)malloc(INSTANCES * ATTRIBUTES * sizeof(char));
  cpu_distance = (int *)malloc(INSTANCES * INSTANCES * sizeof(int));
  gpu_distance = (int *)malloc(INSTANCES * INSTANCES * sizeof(int));

  /* randomly initialize host data */
  #pragma omp parallel for collapse(2)
  for (int i = 0; i < ATTRIBUTES; i++) {
    for (int j = 0; j < INSTANCES; j++) {
      data[i + ATTRIBUTES * j] = data_char[i + ATTRIBUTES * j] = random() % 3;
    }
  }

  /* CPU */
  bzero(cpu_distance,INSTANCES*INSTANCES*sizeof(int));
  gettimeofday(&tp, &tzp);
  start_cpu = tp.tv_sec*1000000+tp.tv_usec;
  CPU(data, cpu_distance);
  gettimeofday(&tp, &tzp);
  stop_cpu = tp.tv_sec*1000000+tp.tv_usec;
  elapsedTime = stop_cpu - start_cpu;
  fprintf(out,"%.2f ",elapsedTime);

#pragma omp target data map(to: data_char[0:INSTANCES * ATTRIBUTES]) \
                        map(alloc: gpu_distance[0:INSTANCES * INSTANCES ])
{

  /* run the register-based kernel 10 times */
  for (int n = 0; n < 10; n++) {
    bzero(gpu_distance,INSTANCES*INSTANCES*sizeof(int));
    gettimeofday(&tp, &tzp);
    start_gpu = tp.tv_sec*1000000+tp.tv_usec;

    #pragma omp target update to (gpu_distance[0:INSTANCES * INSTANCES])
    #pragma omp target teams num_teams(INSTANCES*INSTANCES) thread_limit(THREADS)
    {
      #pragma omp parallel
      {
        int idx = omp_get_thread_num();
        int gx = omp_get_team_num() % INSTANCES;
        int gy = omp_get_team_num() / INSTANCES;
    
        for(int i = 4*idx; i < ATTRIBUTES; i+=THREADS*4) {
          char4 j = *(char4 *)(data_char + i + ATTRIBUTES*gx);
          char4 k = *(char4 *)(data_char + i + ATTRIBUTES*gy);
    
          /* use a local variable (stored in register) to hold intermediate
             values. This reduces writes to global memory */
          char count = 0;
    
          if(j.x ^ k.x) 
            count++; 
          if(j.y ^ k.y)
            count++;
          if(j.z ^ k.z)
            count++;
          if(j.w ^ k.w)
            count++;
    
          /* Only one atomic write to global memory */
          #pragma omp atomic update
          gpu_distance[ INSTANCES*gx + gy ] += count;
        }
      }
    }
    #pragma omp target update from (gpu_distance[0:INSTANCES * INSTANCES])
    gettimeofday(&tp, &tzp);
    stop_gpu = tp.tv_sec*1000000+tp.tv_usec;
    elapsedTime = stop_gpu - start_gpu;
    fprintf(out,"%.2f ",elapsedTime);

  }
  /* check CPU and GPU results */
  status = memcmp(cpu_distance, gpu_distance, INSTANCES * INSTANCES * sizeof(int));
  if (status != 0) printf("FAIL\n");
  else printf("PASS\n");

  /* run the shared-based kernel 10 times */
  for (int n = 0; n < 10; n++) {
    bzero(gpu_distance,INSTANCES*INSTANCES*sizeof(int));
    gettimeofday(&tp, &tzp);
    start_gpu = tp.tv_sec*1000000+tp.tv_usec;

    #pragma omp target update to (gpu_distance[0:INSTANCES * INSTANCES])
    #pragma omp target teams num_teams(INSTANCES*INSTANCES) thread_limit(THREADS)
    {
      int dist[THREADS];
      #pragma omp parallel
      {
        int idx = omp_get_thread_num();
        int gx = omp_get_team_num() % INSTANCES;
        int gy = omp_get_team_num() / INSTANCES;
    
        dist[idx] = 0;
        #pragma omp barrier
    
        for(int i = 4*idx; i < ATTRIBUTES; i+=THREADS*4) {
          char4 j = *(char4 *)(data_char + i + ATTRIBUTES*gx);
          char4 k = *(char4 *)(data_char + i + ATTRIBUTES*gy);
    
          /* use a local variable (stored in register) to hold intermediate
             values. This reduces writes to global memory */
          char count = 0;
    
          if(j.x ^ k.x) 
            count++; 
          if(j.y ^ k.y)
            count++;
          if(j.z ^ k.z)
            count++;
          if(j.w ^ k.w)
            count++;
    
          dist[idx] += count;
        }
    
      /* Synchronize threads to make sure all have completed their updates
         of the shared array. Since the distances for each thread are read
         by thread 0 below, this must be ensured. Above, it was not
         necessary because each thread was accessing its own memory
      */
        #pragma omp barrier
    
      /* Reduction: Thread 0 will add the value of all other threads to
      its own */ 
        if(idx == 0) {
          for(int i = 1; i < THREADS; i++) {
            dist[0] += dist[i];
          }
    
          /* Thread 0 will then write the output to global memory. Note that
             this does not need to be performed atomically, because only one
             thread per block is writing to global memory, and each block
             corresponds to a unique memory address. 
          */
          gpu_distance[INSTANCES*gy + gx] = dist[0];
        }
      }
    }
    #pragma omp target update from (gpu_distance[0:INSTANCES * INSTANCES])
  gettimeofday(&tp, &tzp);
  stop_gpu = tp.tv_sec*1000000+tp.tv_usec;
  elapsedTime = stop_gpu - start_gpu;
  fprintf(out,"%.2f ",elapsedTime);
  }
  /* check CPU and GPU results */
  status = memcmp(cpu_distance, gpu_distance, INSTANCES * INSTANCES * sizeof(int));
  if (status != 0) printf("FAIL\n");
  else printf("PASS\n");
}

  fclose(out);
  free(cpu_distance);
  free(gpu_distance);
  free(data_char);
  free(data);
  return status;
}
 

