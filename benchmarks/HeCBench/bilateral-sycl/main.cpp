#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "common.h"
#include "reference.h"

template<int R>
void bilateralFilter(
    nd_item<2> &item,
    const float *__restrict in,
    float *__restrict out,
    int w, 
    int h, 
    float a_square,
    float variance_I,
    float variance_spatial)
{
  const int idx = item.get_global_id(1);
  const int idy = item.get_global_id(0);

  if(idx >= w || idy >= h) return;

  int id = idy*w + idx;
  float I = in[id];
  float res = 0;
  float normalization = 0;

  // window centered at the coordinate (idx, idy)
#ifdef LOOP_UNROLL
  #pragma unroll
#endif
  for(int i = -R; i <= R; i++) {
#ifdef LOOP_UNROLL
    #pragma unroll
#endif
    for(int j = -R; j <= R; j++) {

      int idk = idx+i;
      int idl = idy+j;

      // mirror edges
      if( idk < 0) idk = -idk;
      if( idl < 0) idl = -idl;
      if( idk > w - 1) idk = w - 1 - i;
      if( idl > h - 1) idl = h - 1 - j;

      int id_w = idl*w + idk;
      float I_w = in[id_w];

      // range kernel for smoothing differences in intensities
      float range = -(I-I_w) * (I-I_w) / (2.f * variance_I);

      // spatial (or domain) kernel for smoothing differences in coordinates
      float spatial = -((idk-idx)*(idk-idx) + (idl-idy)*(idl-idy)) /
                      (2.f * variance_spatial);

      // the weight is assigned using the spatial closeness (using the spatial kernel) 
      // and the intensity difference (using the range kernel)
      float weight = a_square * sycl::exp(spatial + range);

      normalization += weight;
      res += (I_w * weight);
    }
  }
  out[id] = res/normalization;
}

//
// reference https://en.wikipedia.org/wiki/Bilateral_filter
//
int main(int argc, char *argv[]) {

  // image dimensions
  int w = atoi(argv[1]);
  int h = atoi(argv[2]);
  const int img_size = w*h;

   // As the range parameter increases, the bilateral filter gradually 
   // approaches Gaussian convolution more closely because the range 
   // Gaussian widens and flattens, which means that it becomes nearly
   // constant over the intensity interval of the image.
  float variance_I = atof(argv[3]);

   // As the spatial parameter increases, the larger features get smoothened.
  float variance_spatial = atof(argv[4]);

  // square of the height of the curve peak
  float a_square = 0.5f / (variance_I * (float)M_PI);

  float *h_src = (float*) malloc (img_size * sizeof(float));
  // host and device results
  float *h_dst = (float*) malloc (img_size * sizeof(float));
  float *r_dst = (float*) malloc (img_size * sizeof(float));

  srand(123);
  for (int i = 0; i < img_size; i++)
    h_src[i] = rand() % 256;

#ifdef USE_GPU
  gpu_selector dev_sel;
#else
  cpu_selector dev_sel;
#endif
  queue q(dev_sel);

  buffer<float, 1> d_src (h_src, img_size);
  buffer<float, 1> d_dst (img_size);

  range<2> lws (16, 16);
  range<2> gws ((h+15)/16*16, (w+15)/16*16);

  for (int i = 0; i < 100; i++)
    q.submit([&] (handler &cgh) {
      auto src = d_src.get_access<sycl_read>(cgh);
      auto dst = d_dst.get_access<sycl_discard_write>(cgh);
      cgh.parallel_for<class radius3x3>(nd_range<2>(gws, lws), [=] (nd_item<2> item) {
        bilateralFilter<3>(item, src.get_pointer(), dst.get_pointer(), 
                           w, h, a_square, variance_I, variance_spatial);
      });
    });

  q.submit([&] (handler &cgh) {
    auto acc = d_dst.get_access<sycl_read>(cgh);
    cgh.copy(acc, h_dst);
  }).wait();

  // verify
  bool ok = true;
  reference<3>(h_src, r_dst, w, h, a_square, variance_I, variance_spatial);
  for (int i = 0; i < w*h; i++) {
    if (fabsf(r_dst[i] - h_dst[i]) > 1e-3) {
      ok = false;
      break;
    }
  }

  for (int i = 0; i < 100; i++)
    q.submit([&] (handler &cgh) {
      auto src = d_src.get_access<sycl_read>(cgh);
      auto dst = d_dst.get_access<sycl_discard_write>(cgh);
      cgh.parallel_for<class radius6x6>(nd_range<2>(gws, lws), [=] (nd_item<2> item) {
        bilateralFilter<6>(item, src.get_pointer(), dst.get_pointer(), 
                           w, h, a_square, variance_I, variance_spatial);
      });
    });

  q.submit([&] (handler &cgh) {
    auto acc = d_dst.get_access<sycl_read>(cgh);
    cgh.copy(acc, h_dst);
  }).wait();

  reference<6>(h_src, r_dst, w, h, a_square, variance_I, variance_spatial);
  for (int i = 0; i < w*h; i++) {
    if (fabsf(r_dst[i] - h_dst[i]) > 1e-3) {
      ok = false;
      break;
    }
  }

  for (int i = 0; i < 100; i++)
    q.submit([&] (handler &cgh) {
      auto src = d_src.get_access<sycl_read>(cgh);
      auto dst = d_dst.get_access<sycl_discard_write>(cgh);
      cgh.parallel_for<class radius9x9>(nd_range<2>(gws, lws), [=] (nd_item<2> item) {
        bilateralFilter<9>(item, src.get_pointer(), dst.get_pointer(), 
                           w, h, a_square, variance_I, variance_spatial);
      });
    });

  q.submit([&] (handler &cgh) {
    auto acc = d_dst.get_access<sycl_read>(cgh);
    cgh.copy(acc, h_dst);
  }).wait();


  reference<9>(h_src, r_dst, w, h, a_square, variance_I, variance_spatial);
  for (int i = 0; i < w*h; i++) {
    if (fabsf(r_dst[i] - h_dst[i]) > 1e-3) {
      ok = false;
      break;
    }
  }
  printf("%s\n", ok ? "PASS" : "FAIL");

  free(h_dst);
  free(r_dst);
  free(h_src);
  return 0;
};
