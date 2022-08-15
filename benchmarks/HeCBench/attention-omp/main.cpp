#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>

float* attention_host(const float* key, const float* value, const float* query,
                      const int n, const int d) 
{
  // intermediate
  float* dot_product = (float*) malloc (n * sizeof(float));
  float* score = (float*) malloc (n * sizeof(float));
  // result
  float* output = (float*) malloc (d * sizeof(float));

  for (int i = 0; i < n; i++) {
    float sum = 0;
    for (int j = 0; j < d; j++)
       sum += key[i * d + j] * query[j];
    dot_product[i] = sum;
  }

  float sum = 0;
  for (int i = 0; i < n; i++)
    sum += expf(dot_product[i]);

  for (int i = 0; i < n; i++)
    score[i] = expf(dot_product[i]) / sum;
  
  for (int j = 0; j < d; j++) {
    float sum = 0;
    for (int i = 0; i < n; i++)
       sum += score[i] * value[i * d + j];
    output[j] = sum;
  }

  free(dot_product);
  free(score);
  return output;
}

float* attention_device(const float* key, const float* value, const float* query,
                        const int n, const int d, const int repeat) 
{
  // intermediate
  float* dot_product = (float*) malloc (n * sizeof(float));
  float* score = (float*) malloc (n * sizeof(float));
  float* exp_sum = (float*) malloc (sizeof(float));

  // result
  float* output = (float*) malloc (d * sizeof(float));

  #pragma omp target data map(to: key[0:n*d], value[0:n*d], query[0:d]), \
                          map(alloc: dot_product[0:n], score[0:n], exp_sum[0:1]), \
                          map(from: output[0:d])
  {
    for (int k = 0; k < repeat; k++) {
      exp_sum[0] = 0;
      #pragma omp target update to (exp_sum[0:1])

      #pragma omp target teams distribute parallel for thread_limit(256)
      for (int i = 0; i < n; i++) {
        float sum = 0;
        for (int j = 0; j < d; j++)
           sum += key[i * d + j] * query[j];
        dot_product[i] = sum;
        #pragma omp atomic update  
        exp_sum[0] += expf(sum);
      }

      #pragma omp target teams distribute parallel for thread_limit(256)
      for (int i = 0; i < n; i++)
        score[i] = expf(dot_product[i]) / exp_sum[0];
      
      #pragma omp target teams distribute parallel for thread_limit(256)
      for (int j = 0; j < d; j++) {
        float sum = 0;
        for (int i = 0; i < n; i++)
           sum += score[i] * value[i * d + j];
        output[j] = sum;
      }
    }
  }

  free(dot_product);
  free(score);
  free(exp_sum);
  return output;
}

int main(int argc, char* argv[]) {
  const int n = atoi(argv[1]);
  const int d = atoi(argv[2]);
  const int r = atoi(argv[3]);

  // input
  float* key = (float*) malloc (n * d * sizeof(float));
  float* value = (float*) malloc (n * d * sizeof(float));
  float* query = (float*) malloc (d * sizeof(float));

  srand(2);
  for (int i = 0; i < n * d; i++) {
    key[i] = 0.1;
    value[i] = 0.3;
    if (rand() % 2)
      query[i % d] = value[i] + key[i] ;
    else
      query[i % d] = value[i] - key[i] ;
  }

  float* hout = attention_host(key, value, query, n, d);
  float* dout = attention_device(key, value, query, n, d, r);

  float rmse = 0;
  for (int i = 0; i < d; i++) 
    rmse += (hout[i] - dout[i]) * (hout[i] - dout[i]);
  printf("RMSE = %f\n", sqrtf(rmse / d));

  free(key);
  free(value);
  free(query);
  free(dout);
  free(hout);
  return 0;
}
