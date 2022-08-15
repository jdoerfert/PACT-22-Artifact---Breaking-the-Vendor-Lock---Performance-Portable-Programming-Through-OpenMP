/*
MIT License

Copyright (c) 2020 Soumya Sen

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "common.h"

#define idx(i,j)   (i)*y_points+(j)

int main(int argc, char* argv[])
{
  if (argc != 3) {
    printf("Usage: %s dimx dimy\n", argv[0]);
    exit(-1);
  }

  // Define the domain
  const int x_points = atoi(argv[1]);
  const int y_points = atoi(argv[2]);
  const double x_len = 2.0;
  const double y_len = 2.0;
  const double del_x = x_len/(x_points-1);
  const double del_y = y_len/(y_points-1);

  const int grid_elems = x_points * y_points;
  const int grid_size = sizeof(double) * grid_elems;

  double *x = (double*) malloc (sizeof(double) * x_points);
  double *y = (double*) malloc (sizeof(double) * y_points);
  double *u = (double*) malloc (grid_size);
  double *v = (double*) malloc (grid_size);
  double *u_new = (double*) malloc (grid_size);
  double *v_new = (double*) malloc (grid_size);

  // store device results
  double *du = (double*) malloc (grid_size);
  double *dv = (double*) malloc (grid_size);

  // Define the parameters
  const int num_itrs = 100;     // Number of time iterations
  const double nu = 0.01;
  const double sigma = 0.0009;
  const double del_t = sigma * del_x * del_y / nu;      // CFL criteria

  printf("2D Burger's equation\n");
  printf("Grid dimension: x = %d y = %d\n", x_points, y_points);

  for(int i = 0; i < x_points; i++) x[i] = i * del_x;
  for(int i = 0; i < y_points; i++) y[i] = i * del_y;

  for(int i = 0; i < y_points; i++){
    for(int j = 0; j < x_points; j++){
      u[idx(i,j)] = 1.0;
      v[idx(i,j)] = 1.0;
      u_new[idx(i,j)] = 1.0;
      v_new[idx(i,j)] = 1.0;

      if(x[j] > 0.5 && x[j] < 1.0 && y[i] > 0.5 && y[i] < 1.0){
        u[idx(i,j)] = 2.0;
        v[idx(i,j)] = 2.0;
        u_new[idx(i,j)] = 2.0;
        v_new[idx(i,j)] = 2.0;
      }
    }
  }

  { // sycl scope
#ifdef USE_GPU
  gpu_selector dev_sel;
#else
  cpu_selector dev_sel;
#endif
  queue q(dev_sel);

  buffer<double, 1> d_u_new (u_new, grid_elems); 
  buffer<double, 1> d_v_new (v_new, grid_elems);
  buffer<double, 1> d_u (u, grid_elems);
  buffer<double, 1> d_v (v, grid_elems);
  d_u_new.set_final_data(nullptr);
  d_v_new.set_final_data(nullptr);
  d_u.set_final_data(du);
  d_v.set_final_data(dv);

  // ranges of the four kernels
  range<2> gws ((y_points-2+15)/16*16, (x_points-2+15)/16*16);
  range<2> lws (16, 16);
  range<1> gws2 ((x_points+255)/256*256);
  range<1> lws2 (256);
  range<1> gws3 ((y_points+255)/256*256);
  range<1> lws3 (256);
  range<1> gws4 ((grid_elems+255)/256*256);
  range<1> lws4 (256);

  for(int itr = 0; itr < num_itrs; itr++) {

    q.submit([&] (handler &cgh) {
      auto u_new = d_u_new.get_access<sycl_write>(cgh);
      auto v_new = d_v_new.get_access<sycl_write>(cgh);
      auto     u = d_u.get_access<sycl_read>(cgh);
      auto     v = d_v.get_access<sycl_read>(cgh);
      cgh.parallel_for<class core>(nd_range<2>(gws, lws), [=] (nd_item<2> item) {
        int i = item.get_global_id(0) + 1;
        int j = item.get_global_id(1) + 1;
        if (j < x_points - 1 && i < y_points - 1) {
          u_new[idx(i,j)] = u[idx(i,j)] + 
            (nu*del_t/(del_x*del_x)) * (u[idx(i,j+1)] + u[idx(i,j-1)] - 2 * u[idx(i,j)]) + 
            (nu*del_t/(del_y*del_y)) * (u[idx(i+1,j)] + u[idx(i-1,j)] - 2 * u[idx(i,j)]) - 
            (del_t/del_x)*u[idx(i,j)] * (u[idx(i,j)] - u[idx(i,j-1)]) - 
            (del_t/del_y)*v[idx(i,j)] * (u[idx(i,j)] - u[idx(i-1,j)]);

          v_new[idx(i,j)] = v[idx(i,j)] +
            (nu*del_t/(del_x*del_x)) * (v[idx(i,j+1)] + v[idx(i,j-1)] - 2 * v[idx(i,j)]) + 
            (nu*del_t/(del_y*del_y)) * (v[idx(i+1,j)] + v[idx(i-1,j)] - 2 * v[idx(i,j)]) -
            (del_t/del_x)*u[idx(i,j)] * (v[idx(i,j)] - v[idx(i,j-1)]) - 
            (del_t/del_y)*v[idx(i,j)] * (v[idx(i,j)] - v[idx(i-1,j)]);
        }
      });
    });

    // Boundary conditions
    q.submit([&] (handler &cgh) {
      auto u_new = d_u_new.get_access<sycl_write>(cgh);
      auto v_new = d_v_new.get_access<sycl_write>(cgh);
      cgh.parallel_for<class bound_h>(nd_range<1>(gws2, lws2), [=] (nd_item<1> item) {
        int i = item.get_global_id(0);
        if (i < x_points) {
          u_new[idx(0,i)] = 1.0;
          v_new[idx(0,i)] = 1.0;
          u_new[idx(y_points-1,i)] = 1.0;
          v_new[idx(y_points-1,i)] = 1.0;
        }
      });
    });

    q.submit([&] (handler &cgh) {
      auto u_new = d_u_new.get_access<sycl_write>(cgh);
      auto v_new = d_v_new.get_access<sycl_write>(cgh);
      cgh.parallel_for<class bound_v>(nd_range<1>(gws3, lws3), [=] (nd_item<1> item) {
        int j = item.get_global_id(0);
        if (j < y_points) {
          u_new[idx(j,0)] = 1.0;
          v_new[idx(j,0)] = 1.0;
          u_new[idx(j,x_points-1)] = 1.0;
          v_new[idx(j,x_points-1)] = 1.0;
        }
      });
    });

    // Updating older values to newer ones
    q.submit([&] (handler &cgh) {
      auto u_new = d_u_new.get_access<sycl_read>(cgh);
      auto v_new = d_v_new.get_access<sycl_read>(cgh);
      auto u = d_u.get_access<sycl_discard_write>(cgh);
      auto v = d_v.get_access<sycl_discard_write>(cgh);
      cgh.parallel_for<class update>(nd_range<1>(gws4, lws4), [=] (nd_item<1> item) {
        int i = item.get_global_id(0);
        if (i < grid_elems) {
          u[i] = u_new[i];
          v[i] = v_new[i];
        }
      });
    });
  }

  }

  printf("Serial computing for verification...\n");

  // Reset velocities
  for(int i = 0; i < y_points; i++){
    for(int j = 0; j < x_points; j++){
      u[idx(i,j)] = 1.0;
      v[idx(i,j)] = 1.0;
      u_new[idx(i,j)] = 1.0;
      v_new[idx(i,j)] = 1.0;

      if(x[j] > 0.5 && x[j] < 1.0 && y[i] > 0.5 && y[i] < 1.0){
        u[idx(i,j)] = 2.0;
        v[idx(i,j)] = 2.0;
        u_new[idx(i,j)] = 2.0;
        v_new[idx(i,j)] = 2.0;
      }
    }
  }

  for(int itr = 0; itr < num_itrs; itr++){

    for(int i = 1; i < y_points-1; i++){
      for(int j = 1; j < x_points-1; j++){
        u_new[idx(i,j)] = u[idx(i,j)] + (nu*del_t/(del_x*del_x)) * (u[idx(i,j+1)] + u[idx(i,j-1)] - 2 * u[idx(i,j)]) + 
                              (nu*del_t/(del_y*del_y)) * (u[idx(i+1,j)] + u[idx(i-1,j)] - 2 * u[idx(i,j)]) - 
                                 (del_t/del_x)*u[idx(i,j)] * (u[idx(i,j)] - u[idx(i,j-1)]) - 
                                 (del_t/del_y)*v[idx(i,j)] * (u[idx(i,j)] - u[idx(i-1,j)]);

        v_new[idx(i,j)] = v[idx(i,j)] + (nu*del_t/(del_x*del_x)) * (v[idx(i,j+1)] + v[idx(i,j-1)] - 2 * v[idx(i,j)]) + 
                              (nu*del_t/(del_y*del_y)) * (v[idx(i+1,j)] + v[idx(i-1,j)] - 2 * v[idx(i,j)]) -
                                 (del_t/del_x)*u[idx(i,j)] * (v[idx(i,j)] - v[idx(i,j-1)]) - 
                                 (del_t/del_y)*v[idx(i,j)] * (v[idx(i,j)] - v[idx(i-1,j)]);
      }
    }

    // Boundary conditions
    for(int i = 0; i < x_points; i++){
      u_new[idx(0,i)] = 1.0;
      v_new[idx(0,i)] = 1.0;
      u_new[idx(y_points-1,i)] = 1.0;
      v_new[idx(y_points-1,i)] = 1.0;
    }

    for(int j = 0; j < y_points; j++){
      u_new[idx(j,0)] = 1.0;
      v_new[idx(j,0)] = 1.0;
      u_new[idx(j,x_points-1)] = 1.0;
      v_new[idx(j,x_points-1)] = 1.0;
    }

    // Updating older values to newer ones
    for(int i = 0; i < y_points; i++){
      for(int j = 0; j < x_points; j++){
        u[idx(i,j)] = u_new[idx(i,j)];
        v[idx(i,j)] = v_new[idx(i,j)];
      }
    }
  }

  bool ok = true;
  for(int i = 0; i < y_points; i++){
    for(int j = 0; j < x_points; j++){
      if (fabs(du[idx(i,j)] - u[idx(i,j)]) > 1e-6 || 
          fabs(dv[idx(i,j)] - v[idx(i,j)]) > 1e-6) ok = false;
    }
  }
  printf("%s\n", ok ? "PASS" : "FAIL");

  free(x);
  free(y);
  free(u);
  free(v);
  free(du);
  free(dv);
  free(u_new);
  free(v_new);

  return 0;
}
