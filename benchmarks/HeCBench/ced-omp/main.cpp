/*
 * Copyright (c) 2016 University of Cordoba and University of Illinois
 * All rights reserved.
 *
 * Developed by:    IMPACT Research Group
 *                  University of Cordoba and University of Illinois
 *                  http://impact.crhc.illinois.edu/
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * with the Software without restriction, including without limitation the 
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 *      > Redistributions of source code must retain the above copyright notice,
 *        this list of conditions and the following disclaimers.
 *      > Redistributions in binary form must reproduce the above copyright
 *        notice, this list of conditions and the following disclaimers in the
 *        documentation and/or other materials provided with the distribution.
 *      > Neither the names of IMPACT Research Group, University of Cordoba, 
 *        University of Illinois nor the names of its contributors may be used 
 *        to endorse or promote products derived from this Software without 
 *        specific prior written permission.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE 
 * CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS WITH
 * THE SOFTWARE.
 *
 */

#include <unistd.h>
#include <thread>
#include <assert.h>
#include <omp.h>
#include <math.h>

#include "kernel.h"
#include "support/partitioner.h"
#include "support/verify.h"

#define min(a,b) (a) < (b) ? (a) : (b)
#define max(a,b) (a) < (b) ? (b) : (a)

// Params ---------------------------------------------------------------------
struct Params {

  int         device;
  int         n_warmup;
  int         n_reps;
  float       alpha;
  const char *file_name;
  const char *comparison_file;
  int         display = 0;

  Params(int argc, char **argv) {
    device          = 0;
    n_warmup        = 10;
    n_reps          = 100;
    alpha           = 0.2;
    file_name       = "input/peppa/";
    comparison_file = "output/peppa/";
    int opt;
    while((opt = getopt(argc, argv, "hd:i:t:w:r:a:f:c:x")) >= 0) {
      switch(opt) {
        case 'h':
          usage();
          exit(0);
          break;
        case 'd': device          = atoi(optarg); break;
        case 'w': n_warmup        = atoi(optarg); break;
        case 'r': n_reps          = atoi(optarg); break;
        case 'a': alpha           = atof(optarg); break;
        case 'f': file_name       = optarg; break;
        case 'c': comparison_file = optarg; break;
        case 'x': display         = 1; break;
        default:
            fprintf(stderr, "\nUnrecognized option!\n");
            usage();
            exit(0);
      }
    }
#ifndef CHAI_OPENCV
    assert(display != 1 && "Compile with CHAI_OPENCV");
#endif
  }

  void usage() {
    fprintf(stderr,
        "\nUsage:  ./ced [options]"
        "\n"
        "\nGeneral options:"
        "\n    -h        help"
        "\n    -t <T>    # of host threads (default=4)"
        "\n    -w <W>    # of untimed warmup iterations (default=10)"
        "\n    -r <R>    # of timed repetition iterations (default=100)"
        "\n"
        "\nBenchmark-specific options:"
        "\n    -f <F>    folder containing input video files (default=input/peppa/)"
        "\n    -c <C>    folder containing comparison files (default=output/peppa/)"
        "\n    -x        display output video (with CHAI_OPENCV)"
        "\n");
  }
};

// Input Data -----------------------------------------------------------------
void read_input(unsigned char** all_gray_frames, 
    int &rows, int &cols, int &in_size, const Params &p) {

  for(int task_id = 0; task_id < p.n_warmup + p.n_reps; task_id++) {

    char FileName[100];
    sprintf(FileName, "%s%d.txt", p.file_name, task_id);

    FILE *fp = fopen(FileName, "r");
    if(fp == NULL) {
      perror ("The following error occurred");
      exit(EXIT_FAILURE);
    }

    fscanf(fp, "%d\n", &rows);
    fscanf(fp, "%d\n", &cols);

    in_size = rows * cols * sizeof(unsigned char);
    all_gray_frames[task_id]    = (unsigned char *)malloc(in_size);
    for(int i = 0; i < rows; i++) {
      for(int j = 0; j < cols; j++) {
        fscanf(fp, "%u ", (unsigned int *)&all_gray_frames[task_id][i * cols + j]);
      }
    }
    fclose(fp);
  }
}

// Main ------------------------------------------------------------------------------------------
int main(int argc, char **argv) {

  Params      p(argc, argv);

  // The maximum number of GPU threads is 1024 for certain GPUs
  const int max_gpu_threads = 256;

  // read data from an 'input' directory which must be available
  const int n_frames = p.n_warmup + p.n_reps;
  unsigned char **all_gray_frames = (unsigned char **)malloc(n_frames * sizeof(unsigned char *));
  int     rows, cols, in_size;
  read_input(all_gray_frames, rows, cols, in_size, p);

  unsigned char **all_out_frames = (unsigned char **)malloc(n_frames * sizeof(unsigned char *));
  for(int i = 0; i < n_frames; i++) {
    all_out_frames[i] = (unsigned char *)malloc(in_size);
  }

  unsigned char* gpu_in_out = (unsigned char *)malloc(in_size);

  unsigned char *interm_gpu_proxy = (unsigned char *)malloc(in_size);
  unsigned char *theta_gpu_proxy  = (unsigned char *)malloc(in_size);

#pragma omp target data map(alloc: gpu_in_out[0:in_size], \
    interm_gpu_proxy[0:in_size], \
    theta_gpu_proxy[0:in_size]) 
  {

    for(int task_id = 0; task_id < n_frames; task_id++) {

      // Next frame
      memcpy(gpu_in_out, all_gray_frames[task_id], in_size);


      // Copy to Device

#pragma omp target update to (gpu_in_out[0:in_size])

      const int threads = 16;
      int team_size = (rows-2)/threads*(cols-2)/threads;

      // call GAUSSIAN KERNEL
#pragma omp target teams num_teams(team_size) thread_limit(max_gpu_threads)
      {

        int l_data[(threads+2)*(threads+2)];

#pragma omp parallel
        {
          const int L_SIZE = 16; //threads; 
          const int l_row = omp_get_thread_num() / L_SIZE + 1;
          const int l_col = omp_get_thread_num() % L_SIZE + 1;
          const int g_row = omp_get_team_num() / ((cols - 2) / L_SIZE) * L_SIZE + l_row;
          const int g_col = omp_get_team_num() % ((cols - 2) / L_SIZE) * L_SIZE + l_col;
          const int pos = g_row * cols + g_col;
          int sum         = 0;

          const float gaus[9] = {0.0625f, 0.125f, 0.0625f, 
            0.1250f, 0.250f, 0.1250f, 
            0.0625f, 0.125f, 0.0625f};

          // copy to local
          l_data[l_row * (L_SIZE + 2) + l_col] = gpu_in_out[pos];

          // top most row
          if(l_row == 1) {
            l_data[0 * (L_SIZE + 2) + l_col] = gpu_in_out[pos - cols];
            // top left
            if(l_col == 1)
              l_data[0 * (L_SIZE + 2) + 0] = gpu_in_out[pos - cols - 1];

            // top right
            else if(l_col == L_SIZE)
              l_data[0 * (L_SIZE + 2) + L_SIZE + 1] = gpu_in_out[pos - cols + 1];
          }
          // bottom most row
          else if(l_row == L_SIZE) {
            l_data[(L_SIZE + 1) * (L_SIZE + 2) + l_col] = gpu_in_out[pos + cols];
            // bottom left
            if(l_col == 1)
              l_data[(L_SIZE + 1) * (L_SIZE + 2) + 0] = gpu_in_out[pos + cols - 1];

            // bottom right
            else if(l_col == L_SIZE)
              l_data[(L_SIZE + 1) * (L_SIZE + 2) + L_SIZE + 1] = gpu_in_out[pos + cols + 1];
          }

          if(l_col == 1)
            l_data[l_row * (L_SIZE + 2) + 0] = gpu_in_out[pos - 1];
          else if(l_col == L_SIZE)
            l_data[l_row * (L_SIZE + 2) + L_SIZE + 1] = gpu_in_out[pos + 1];

#pragma omp barrier

          for(int i = 0; i < 3; i++) {
            for(int j = 0; j < 3; j++) {
              sum += gaus[i*3+j] * l_data[(i + l_row - 1) * (L_SIZE + 2) + j + l_col - 1];
            }
          }
          interm_gpu_proxy[pos] = min(255, max(0, sum));
        }
      }

      // call SOBEL KERNEL
#pragma omp target teams num_teams(team_size) thread_limit(max_gpu_threads)
      {

        int l_data[(threads+2)*(threads+2)];

#pragma omp parallel 
        {
          const int L_SIZE = 16;
          const int l_row = omp_get_thread_num() / L_SIZE + 1;
          const int l_col = omp_get_thread_num() % L_SIZE + 1;
          const int g_row = omp_get_team_num() / ((cols - 2) / L_SIZE) * L_SIZE + l_row;
          const int g_col = omp_get_team_num() % ((cols - 2) / L_SIZE) * L_SIZE + l_col;
          const int pos = g_row * cols + g_col;
          const float PI    = 3.14159265f;
          const int   sobx[9] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
          const int   soby[9] = {-1, -2, -1, 0, 0, 0, 1, 2, 1};

          // copy to local
          l_data[l_row * (L_SIZE + 2) + l_col] = interm_gpu_proxy[pos];

          // top most row
          if(l_row == 1) {
            l_data[0 * (L_SIZE + 2) + l_col] = interm_gpu_proxy[pos - cols];
            // top left
            if(l_col == 1)
              l_data[0 * (L_SIZE + 2) + 0] = interm_gpu_proxy[pos - cols - 1];

            // top right
            else if(l_col == L_SIZE)
              l_data[0 * (L_SIZE + 2) + (L_SIZE + 1)] = interm_gpu_proxy[pos - cols + 1];
          }
          // bottom most row
          else if(l_row == L_SIZE) {
            l_data[(L_SIZE + 1) * (L_SIZE + 2) + l_col] = interm_gpu_proxy[pos + cols];
            // bottom left
            if(l_col == 1)
              l_data[(L_SIZE + 1) * (L_SIZE + 2) + 0] = interm_gpu_proxy[pos + cols - 1];

            // bottom right
            else if(l_col == L_SIZE)
              l_data[(L_SIZE + 1) * (L_SIZE + 2) + (L_SIZE + 1)] = interm_gpu_proxy[pos + cols + 1];
          }

          // left
          if(l_col == 1)
            l_data[l_row * (L_SIZE + 2) + 0] = interm_gpu_proxy[pos - 1];
          // right
          else if(l_col == L_SIZE)
            l_data[l_row * (L_SIZE + 2) + (L_SIZE + 1)] = interm_gpu_proxy[pos + 1];

#pragma omp barrier

          float sumx = 0, sumy = 0, angle = 0;
          // find x and y derivatives
          for(int i = 0; i < 3; i++) {
            for(int j = 0; j < 3; j++) {
              sumx += sobx[i*3+j] * l_data[(i + l_row - 1) * (L_SIZE + 2) + j + l_col - 1];
              sumy += soby[i*3+j] * l_data[(i + l_row - 1) * (L_SIZE + 2) + j + l_col - 1];
            }
          }

          // The output is now the square root of their squares, but they are
          // constrained to 0 <= value <= 255. Note that hypot is a built in function
          // defined as: hypot(x,y) = sqrt(x*x, y*y).
          gpu_in_out[pos] = min(255, max(0, (int)hypotf(sumx, sumy)));

          // Compute the direction angle theta_gpu_proxy in radians
          // atan2 has a range of (-PI, PI) degrees
          angle = atan2f(sumy, sumx);

          // If the angle is negative,
          // shift the range to (0, 2PI) by adding 2PI to the angle,
          // then perform modulo operation of 2PI
          if(angle < 0) {
            angle = fmodf((angle + 2 * PI), (2 * PI));
          }

          // Round the angle to one of four possibilities: 0, 45, 90, 135 degrees
          // then store it in the theta_gpu_proxy buffer at the proper position
          //theta_gpu_proxy[pos] = ((int)(degrees(angle * (PI/8) + PI/8-0.0001) / 45) * 45) % 180;
          if(angle <= PI / 8)
            theta_gpu_proxy[pos] = 0;
          else if(angle <= 3 * PI / 8)
            theta_gpu_proxy[pos] = 45;
          else if(angle <= 5 * PI / 8)
            theta_gpu_proxy[pos] = 90;
          else if(angle <= 7 * PI / 8)
            theta_gpu_proxy[pos] = 135;
          else if(angle <= 9 * PI / 8)
            theta_gpu_proxy[pos] = 0;
          else if(angle <= 11 * PI / 8)
            theta_gpu_proxy[pos] = 45;
          else if(angle <= 13 * PI / 8)
            theta_gpu_proxy[pos] = 90;
          else if(angle <= 15 * PI / 8)
            theta_gpu_proxy[pos] = 135;
          else
            theta_gpu_proxy[pos] = 0; // (angle <= 16*PI/8)
        }
      }

      // call NON-MAXIMUM SUPPRESSION KERNEL
#pragma omp target teams num_teams(team_size) thread_limit(max_gpu_threads)
      {

        int l_data[(threads+2)*(threads+2)];

#pragma omp parallel 
        {
          const int L_SIZE = 16;
          const int l_row = omp_get_thread_num() / L_SIZE + 1;
          const int l_col = omp_get_thread_num() % L_SIZE + 1;
          const int g_row = omp_get_team_num() / ((cols - 2) / L_SIZE) * L_SIZE + l_row;
          const int g_col = omp_get_team_num() % ((cols - 2) / L_SIZE) * L_SIZE + l_col;
          const int pos = g_row * cols + g_col;

          // copy to l_data
          l_data[l_row * (L_SIZE + 2) + l_col] = gpu_in_out[pos];

          // top most row
          if(l_row == 1) {
            l_data[0 * (L_SIZE + 2) + l_col] = gpu_in_out[pos - cols];
            // top left
            if(l_col == 1)
              l_data[0 * (L_SIZE + 2) + 0] = gpu_in_out[pos - cols - 1];

            // top right
            else if(l_col == L_SIZE)
              l_data[0 * (L_SIZE + 2) + (L_SIZE + 1)] = gpu_in_out[pos - cols + 1];
          }
          // bottom most row
          else if(l_row == L_SIZE) {
            l_data[(L_SIZE + 1) * (L_SIZE + 2) + l_col] = gpu_in_out[pos + cols];
            // bottom left
            if(l_col == 1)
              l_data[(L_SIZE + 1) * (L_SIZE + 2) + 0] = gpu_in_out[pos + cols - 1];

            // bottom right
            else if(l_col == L_SIZE)
              l_data[(L_SIZE + 1) * (L_SIZE + 2) + (L_SIZE + 1)] = gpu_in_out[pos + cols + 1];
          }

          if(l_col == 1)
            l_data[l_row * (L_SIZE + 2) + 0] = gpu_in_out[pos - 1];
          else if(l_col == L_SIZE)
            l_data[l_row * (L_SIZE + 2) + (L_SIZE + 1)] = gpu_in_out[pos + 1];

#pragma omp barrier

          unsigned char my_magnitude = l_data[l_row * (L_SIZE + 2) + l_col];

          // The following variables are used to address the matrices more easily
          switch(theta_gpu_proxy[pos]) {
            // A gradient angle of 0 degrees = an edge that is North/South
            // Check neighbors to the East and West
            case 0:
              // supress me if my neighbor has larger magnitude
              if(my_magnitude <= l_data[l_row * (L_SIZE + 2) + l_col + 1] || // east
                  my_magnitude <= l_data[l_row * (L_SIZE + 2) + l_col - 1]) // west
              {
                interm_gpu_proxy[pos] = 0;
              }
              // otherwise, copy my value to the output buffer
              else {
                interm_gpu_proxy[pos] = my_magnitude;
              }
              break;

              // A gradient angle of 45 degrees = an edge that is NW/SE
              // Check neighbors to the NE and SW
            case 45:
              // supress me if my neighbor has larger magnitude
              if(my_magnitude <= l_data[(l_row - 1) * (L_SIZE + 2) + l_col + 1] || // north east
                  my_magnitude <= l_data[(l_row + 1) * (L_SIZE + 2) + l_col - 1]) // south west
              {
                interm_gpu_proxy[pos] = 0;
              }
              // otherwise, copy my value to the output buffer
              else {
                interm_gpu_proxy[pos] = my_magnitude;
              }
              break;

              // A gradient angle of 90 degrees = an edge that is E/W
              // Check neighbors to the North and South
            case 90:
              // supress me if my neighbor has larger magnitude
              if(my_magnitude <= l_data[(l_row - 1) * (L_SIZE + 2) + l_col] || // north
                  my_magnitude <= l_data[(l_row + 1) * (L_SIZE + 2) + l_col]) // south
              {
                interm_gpu_proxy[pos] = 0;
              }
              // otherwise, copy my value to the output buffer
              else {
                interm_gpu_proxy[pos] = my_magnitude;
              }
              break;

              // A gradient angle of 135 degrees = an edge that is NE/SW
              // Check neighbors to the NW and SE
            case 135:
              // supress me if my neighbor has larger magnitude
              if(my_magnitude <= l_data[(l_row - 1) * (L_SIZE + 2) + l_col - 1] || // north west
                  my_magnitude <= l_data[(l_row + 1) * (L_SIZE + 2) + l_col + 1]) // south east
              {
                interm_gpu_proxy[pos] = 0;
              }
              // otherwise, copy my value to the output buffer
              else {
                interm_gpu_proxy[pos] = my_magnitude;
              }
              break;

            default: interm_gpu_proxy[pos] = my_magnitude; break;
          }
        }
      }

      // call HYSTERESIS KERNEL
#pragma omp target teams num_teams(team_size) thread_limit(max_gpu_threads)
      {

#pragma omp parallel 
        {
          const int l_row = omp_get_thread_num() / 16 + 1;
          const int l_col = omp_get_thread_num() % 16 + 1;
          const int g_row = omp_get_team_num() / ((cols - 2) / 16) * 16 + l_row;
          const int g_col = omp_get_team_num() % ((cols - 2) / 16) * 16 + l_col;
          const int pos = g_row * cols + g_col;
          float lowThresh  = 10;
          float highThresh = 70;
          const unsigned char EDGE = 255;

          unsigned char magnitude = interm_gpu_proxy[pos];

          if(magnitude >= highThresh)
            gpu_in_out[pos] = EDGE;
          else if(magnitude <= lowThresh)
            gpu_in_out[pos] = 0;
          else {
            float med = (highThresh + lowThresh) / 2;

            if(magnitude >= med)
              gpu_in_out[pos] = EDGE;
            else
              gpu_in_out[pos] = 0;
          }
        }
      }

#pragma omp target update from(gpu_in_out[0:in_size])

      memcpy(all_out_frames[task_id], gpu_in_out, in_size);

    }

  } // #pragma omp target

#ifdef CHAI_OPENCV
  // Display the result
  if(p.display){
    for(int rep = 0; rep < p.n_warmup + p.n_reps; rep++) {
      cv::Mat out_frame = cv::Mat(rows, cols, CV_8UC1);
      memcpy(out_frame.data, all_out_frames[rep], in_size);
      if(!out_frame.empty())
        imshow("canny", out_frame);
      if(cv::waitKey(30) >= 0)
        break;
    }
  }
#endif

  // Verify answer
  int status = verify(all_out_frames, in_size, p.comparison_file, 
      p.n_warmup + p.n_reps, rows, cols, rows, cols);

  // Release buffers
  free(gpu_in_out);
  free(interm_gpu_proxy);
  free(theta_gpu_proxy);
  for(int i = 0; i < n_frames; i++) {
    free(all_gray_frames[i]);
  }
  free(all_gray_frames);
  for(int i = 0; i < n_frames; i++) {
    free(all_out_frames[i]);
  }
  free(all_out_frames);

  if (status == 0) printf("Test Passed\n");
  return 0;
}
