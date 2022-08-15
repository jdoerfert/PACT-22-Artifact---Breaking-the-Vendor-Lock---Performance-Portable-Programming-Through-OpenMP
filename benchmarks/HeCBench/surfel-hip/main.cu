#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <hip/hip_runtime.h>

#define COL_P_X 0
#define COL_P_Y 1
#define COL_P_Z 2
#define COL_N_X 3
#define COL_N_Y 4
#define COL_N_Z 5
#define COL_RSq 6
#define COL_DIM 7

// compute the xyz images using the inverse focal length invF
template<typename T>
__global__ void surfel_render(
  const T *__restrict__ s,
  int N,
  T f,
  int w,
  int h,
  T *__restrict__ d)
{
  const int idx = threadIdx.x + blockIdx.x*blockDim.x;
  const int idy = threadIdx.y + blockIdx.y*blockDim.y;

  if(idx < w && idy < h)
  {
    T ray[3];
    ray[0] = T(idx)-(w-1)*(T)0.5;
    ray[1] = T(idy)-(h-1)*(T)0.5;
    ray[2] = f;
    T pt[3];
    T n[3];
    T p[3];
    T dMin = 1e20;
    
    for (int i=0; i<N; ++i) {
      p[0] = s[i*COL_DIM+COL_P_X];
      p[1] = s[i*COL_DIM+COL_P_Y];
      p[2] = s[i*COL_DIM+COL_P_Z];
      n[0] = s[i*COL_DIM+COL_N_X];
      n[1] = s[i*COL_DIM+COL_N_Y];
      n[2] = s[i*COL_DIM+COL_N_Z];
      T rSqMax = s[i*COL_DIM+COL_RSq];
      T pDotn = p[0]*n[0]+p[1]*n[1]+p[2]*n[2];
      T dsDotRay = ray[0]*n[0] + ray[1]*n[1] + ray[2]*n[2];
      T alpha = pDotn / dsDotRay;
      pt[0] = ray[0]*alpha - p[0];
      pt[1] = ray[1]*alpha - p[1];
      pt[2] = ray[2]*alpha - p[2];
      T t = ray[2]*alpha;
      T rSq = pt[0] * pt[0] + pt[1] * pt[1] + pt[2] * pt[2];
      if (rSq < rSqMax && dMin > t) {
        dMin = t; // ray hit the surfel 
      }
    }
    d[idy*w+idx] = dMin > (T)100 ? (T)0 : dMin;
  }
}

template <typename T>
void surfelRenderTest(int n, int w, int h)
{
  const int src_size = n*7;
  const int dst_size = w*h;

  T *d_src, *d_dst;
  hipMalloc((void**)&d_dst, dst_size * sizeof(T));
  hipMalloc((void**)&d_src, src_size * sizeof(T));

  T *h_dst = (T*) malloc (dst_size * sizeof(T));
  T *h_src = (T*) malloc (src_size * sizeof(T));

  srand(123);
  for (int i = 0; i < src_size; i++)
    h_src[i] = rand() % 256;

  T inverseFocalLength[3] = {0.005, 0.02, 0.036};

  hipMemcpy(d_src, h_src, src_size * sizeof(T), hipMemcpyHostToDevice); 

  dim3 threads(16, 16);
  dim3 blocks((w+15)/16, (h+15)/16);
  for (int f = 0; f < 3; f++) {
    for (int i = 0; i < 100; i++)
      hipLaunchKernelGGL(HIP_KERNEL_NAME(surfel_render<T>), blocks, threads, 
                         0, 0, d_src, n, inverseFocalLength[f], w, h, d_dst);

    hipMemcpy(h_dst, d_dst, dst_size * sizeof(T), hipMemcpyDeviceToHost); 
    T *min = std::min_element( h_dst, h_dst + w*h );
    T *max = std::max_element( h_dst, h_dst + w*h );
    printf("value range [%e, %e]\n", *min, *max);
  }

  free(h_dst);
  free(h_src);
  hipFree(d_dst);
  hipFree(d_src);
}

int main(int argc, char *argv[]) {
  int n = atoi(argv[1]);
  int w = atoi(argv[2]);
  int h = atoi(argv[3]);
  surfelRenderTest<float>(n, w, h);
  surfelRenderTest<double>(n, w, h);
  return 0;
}
