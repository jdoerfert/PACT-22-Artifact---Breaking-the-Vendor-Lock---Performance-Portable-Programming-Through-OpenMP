/* 
 * NMS Benchmarking Framework
 *
 * "Work-Efficient Parallel Non-Maximum Suppression Kernels"
 * Copyright (c) 2019 David Oro et al.
 * 
 * This program is free software: you can redistribute it and/or modify  
 * it under the terms of the GNU General Public License as published by  
 * the Free Software Foundation, version 3.
 *
 * This program is distributed in the hope that it will be useful, but 
 * WITHOUT ANY WARRANTY; without even the implied warranty of 
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU 
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License 
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <omp.h>

typedef struct { float x; float y; float z; float w; } float4 ;

#define MAX_DETECTIONS  4096
#define N_PARTITIONS    32
#define ITER            100

void print_help()
{
  printf("\nUsage: nmstest  <detections.txt>  <output.txt>\n\n");
  printf("               detections.txt -> Input file containing the coordinates, width, and scores of detected objects\n");
  printf("               output.txt     -> Output file after performing NMS\n\n");
}


/* Gets the optimal X or Y dimension for a given CUDA block */
int get_optimal_dim(int val)
{
  int div, neg, cntneg, cntpos;


  /* We start figuring out if 'val' is divisible by 16 
     (e.g. optimal 16x16 CUDA block of maximum GPU occupancy */

  neg = 1;
  div = 16;
  cntneg = div;
  cntpos = div;

  /* In order to guarantee the ending of this loop if 'val' is 
     a prime number, we limit the loop to 5 iterations */

  for(int i=0; i<5; i++)
  {
    if(val % div == 0)
      return div;

    if(neg)
    {
      cntneg--;
      div = cntneg;
      neg = 0;
    }

    else
    {
      cntpos++;
      div = cntpos;
      neg = 1;
    }
  }

  return 16;
}


/* Gets an upper limit for 'val' multiple of the 'mul' integer */
int get_upper_limit(int val, int mul)
{
  int cnt = mul;

  /* The upper limit must be lower than
     the maximum allowed number of detections */

  while(cnt < val)
    cnt += mul;

  if(cnt > MAX_DETECTIONS)
    cnt = MAX_DETECTIONS;

  return cnt;
}

int main(int argc, char *argv[])
{
  int x, y, w;
  float score;

  if(argc != 3)
  {
    print_help();
    return 0;
  }

  /* Read input detection coordinates from the text file */
  int ndetections = 0;

  FILE *fp = fopen(argv[1], "r");
  if (!fp)
  {
    printf("Error: Unable to open file %s for input detection coordinates.\n", argv[1]);
    return -1;
  }

  /* Memory allocation in the host memory address space */
  float4* points = (float4*) malloc(sizeof(float4) * MAX_DETECTIONS);
  if(!points)
  {
    printf("Error: Unable to allocate CPU memory.\n");
    return -1;
  }

  memset(points, 0, sizeof(float4) * MAX_DETECTIONS);

  while(!feof(fp))
  {
    int cnt = fscanf(fp, "%d,%d,%d,%f\n", &x, &y, &w, &score);

    if (cnt !=4)
    {
      printf("Error: Invalid file format in line %d when reading %s\n", ndetections, argv[1]);
      return -1;
    }

    points[ndetections].x = (float) x;       // x coordinate
    points[ndetections].y = (float) y;       // y coordinate
    points[ndetections].z = (float) w;       // window dimensions
    points[ndetections].w = score;           // score

    ndetections++;
  }

  printf("Number of detections read from input file (%s): %d\n", argv[1], ndetections);

  fclose(fp);

  /* CPU array for storing the detection bitmap */
  unsigned char* pointsbitmap = (unsigned char*) malloc(sizeof(unsigned char) * MAX_DETECTIONS);
  memset(pointsbitmap, 0, sizeof(unsigned char) * MAX_DETECTIONS);

  unsigned char* nmsbitmap = (unsigned char*) malloc(sizeof(unsigned char) * MAX_DETECTIONS * MAX_DETECTIONS);
  memset(nmsbitmap, 1, sizeof(unsigned char) * MAX_DETECTIONS * MAX_DETECTIONS);

  /* We build up the non-maximum supression bitmap matrix by removing overlapping windows */
  const int limit = get_upper_limit(ndetections, 16);
  const int threads = get_optimal_dim(limit) * get_optimal_dim(limit);

  #pragma omp target data map(to: points[0:MAX_DETECTIONS], \
                                  nmsbitmap[0:MAX_DETECTIONS * MAX_DETECTIONS]) \
                          map(tofrom: pointsbitmap[0:MAX_DETECTIONS])
  {
    for (int n = 0; n < ITER; n++) {
      #pragma omp target teams distribute parallel for collapse(2) thread_limit(threads)
      for (int i = 0; i < limit; i++) {
        for (int j = 0; j < limit; j++) {
          if(points[i].w < points[j].w)
          {
            float area = (points[j].z + 1.0f) * (points[j].z + 1.0f);
            float w = fmaxf(0.0f, fminf(points[i].x + points[i].z, points[j].x + points[j].z) - 
                fmaxf(points[i].x, points[j].x) + 1.0f);
            float h = fmaxf(0.0f, fminf(points[i].y + points[i].z, points[j].y + points[j].z) - 
                fmaxf(points[i].y, points[j].y) + 1.0f);
            nmsbitmap[i * MAX_DETECTIONS + j] = (((w * h) / area) < 0.3f) && (points[j].z != 0);
          }
        }
      }
    }

    /* Then we perform a reduction for generating a point bitmap vector */
    for (int n = 0; n < ITER; n++) {
#pragma omp target teams num_teams(ndetections) thread_limit(MAX_DETECTIONS / N_PARTITIONS)
      {
#ifdef BYTE_ATOMIC
        unsigned char s;
#else
        unsigned s;
#endif
#pragma omp parallel 
        {
          int bid = omp_get_team_num();
          int lid = omp_get_thread_num();
          int idx = bid * MAX_DETECTIONS + lid;

          if (lid == 0) s = 1;
#pragma omp barrier

#pragma omp atomic update  
          s &= 
#ifndef BYTE_ATOMIC
            (unsigned int)
#endif
            nmsbitmap[idx];
#pragma omp barrier

          for(int i=0; i<(N_PARTITIONS-1); i++)
          {
            idx += MAX_DETECTIONS / N_PARTITIONS;
#pragma omp atomic update  
            s &= nmsbitmap[idx];
#pragma omp barrier
          }
          pointsbitmap[bid] = s;
        }
      }
    }
  }


  fp = fopen(argv[2], "w");
  if (!fp)
  {
    printf("Error: Unable to open file %s for detection outcome.\n", argv[2]);
    return -1;
  }

  int totaldets = 0;
  for(int i = 0; i < ndetections; i++)
  {
    if(pointsbitmap[i])
    {
      x = (int) points[i].x;          // x coordinate
      y = (int) points[i].y;          // y coordinate
      w = (int) points[i].z;          // window dimensions
      score = points[i].w;            // score
      fprintf(fp, "%d,%d,%d,%f\n", x, y, w, score);
      totaldets++; 
    }
  }
  fclose(fp);
  printf("Detections after NMS: %d\n", totaldets);

  free(points);
  free(pointsbitmap);
  free(nmsbitmap);

  return 0;
}

