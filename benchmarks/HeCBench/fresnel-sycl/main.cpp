#include <stdlib.h>
#include <stdio.h>
#include "common.h"

SYCL_EXTERNAL
double Fresnel_Sine_Integral(double);

void reference (const double *__restrict input, double *__restrict output, const int n) {
  for (int i = 0; i < n; i++)
    output[i] = Fresnel_Sine_Integral(input[i]);
}

int main() {
  // range [0, 8], interval 1e-7
  const double interval = 1e-7;
  const int points = (int)(8.0 / interval);
  double *x = (double*) malloc (sizeof(double) * points);
  double *output = (double*) malloc (sizeof(double) * points);
  double *h_output = (double*) malloc (sizeof(double) * points);
  for (int i = 0; i < points; i++)
    x[i] = (double)i * interval;
	   
  { // sycl scope
#ifdef USE_GPU
  gpu_selector dev_sel;
#else
  cpu_selector dev_sel;
#endif

  queue q(dev_sel);
  buffer<double, 1> d_x (x, points);
  buffer<double, 1> d_output (output, points);

  range<1> gws ((points + 255)/256*256);
  range<1> lws (256);

  for (int i = 0; i < 100; i++)
    q.submit([&] (handler &cgh) {
      auto input = d_x.get_access<sycl_read>(cgh);
      auto output = d_output.get_access<sycl_discard_write>(cgh);
      cgh.parallel_for<class fresnel>(nd_range<1>(gws, lws), [=] (nd_item<1> item) {
        int i = item.get_global_id(0);
        if (i < points) output[i] = Fresnel_Sine_Integral(input[i]);
      });
    });
  }

  // verify
  reference(x, h_output, points);
  bool ok = true;
  for (int i = 0; i < points; i++) {
    if (fabs(h_output[i] - output[i]) > 1e-6) {
      printf("%lf %lf\n", h_output[i], output[i]);
      ok = false;
      break;
    }
  }
  printf("%s\n", ok ? "PASS" : "FAIL");
  
  free(x);
  free(output);
  free(h_output);
  return 0;
}
