#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include "reference.h"

// limits of integration
#define A 0
#define B 15

// row size is related to accuracy
#define ROW_SIZE 17
#define EPS      1e-7

inline double f(double x)
{
  return exp(x)*sin(x);
}

inline unsigned int getFirstSetBitPos(int n)
{
  return log2((float)(n&-n))+1;
}

int main( int argc, char** argv)
{
  const int numBlocks = 128;
  const int numThreadsPerBlock = 64;

  double *result = (double*) malloc (sizeof(double) * numBlocks);

  double d_sum;
  double a = A;
  double b = B;
  #pragma omp target data map (from: result[0:numBlocks])
  {
    for (int i = 0; i < 1; i++) {
      #pragma omp target teams num_teams(numBlocks) thread_limit(numThreadsPerBlock) 
      {
        double smem[ROW_SIZE * 64];
        #pragma omp parallel
        {
          int threadIdx_x = omp_get_thread_num();
          int blockIdx_x = omp_get_team_num();
          int gridDim_x = omp_get_num_teams();
          int blockDim_x = omp_get_num_threads();
          double diff = (b-a)/gridDim_x, step;
          int k;
          int max_eval = (1<<(ROW_SIZE-1));
          b = a + (blockIdx_x+1)*diff;
          a += blockIdx_x*diff;

          step = (b-a)/max_eval;

          double local_col[ROW_SIZE];  // specific to the row size
          for(int i = 0; i < ROW_SIZE; i++) local_col[i] = 0.0;
          if(!threadIdx_x)
          {
            k = blockDim_x;
            local_col[0] = f(a) + f(b);
          }
          else
            k = threadIdx_x;

          for(; k < max_eval; k += blockDim_x)
            local_col[ROW_SIZE - getFirstSetBitPos(k)] += 2.0*f(a + step*k);

          for(int i = 0; i < ROW_SIZE; i++)
            smem[ROW_SIZE*threadIdx_x + i] = local_col[i];
          #pragma omp barrier

          if(threadIdx_x < ROW_SIZE)
          {
            double sum = 0.0;
            for(int i = threadIdx_x; i < blockDim_x*ROW_SIZE; i+=ROW_SIZE)
              sum += smem[i];
            smem[threadIdx_x] = sum;
          }

          if(!threadIdx_x)
          {
            double *table = local_col;
            table[0] = smem[0];

            for(int k = 1; k < ROW_SIZE; k++)
              table[k] = table[k-1] + smem[k];

            for(int k = 0; k < ROW_SIZE; k++)  
              table[k]*= (b-a)/(1<<(k+1));

            for(int col = 0 ; col < ROW_SIZE-1 ; col++)
              for(int row = ROW_SIZE-1; row > col; row--)
                table[row] = table[row] + (table[row] - table[row-1])/((1<<(2*col+1))-1);

            result[blockIdx_x] = table[ROW_SIZE-1];
          }
        }
      }
      #pragma omp target update from (result[0:numBlocks])
      d_sum = 0.0;
      for(int k = 0; k < numBlocks; k++) d_sum += result[k];
    }
  }

  // verify
  double ref_sum = reference(f, A, B, ROW_SIZE, EPS);
  printf("%s\n", (fabs(d_sum - ref_sum) > EPS) ? "FAIL" : "PASS");

  free(result);
  return 0;
}

