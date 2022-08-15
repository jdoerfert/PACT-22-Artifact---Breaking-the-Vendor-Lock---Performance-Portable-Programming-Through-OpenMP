#include "common.h"
#include "utils.h"


int main(int argc, char* argv[]) {

  // print timing info if timing is non-zero
  int timing = atoi(argv[1]);

  DATA_TYPE *A = (DATA_TYPE*)malloc(MAP_SIZE * MAP_SIZE * sizeof(DATA_TYPE));
  DATA_TYPE *B = (DATA_TYPE*)malloc((MAP_SIZE - 2) * (MAP_SIZE - 2) * sizeof(DATA_TYPE));
  DATA_TYPE *B_outputFromGpu = (DATA_TYPE*)malloc((MAP_SIZE - 2) * (MAP_SIZE - 2) * sizeof(DATA_TYPE));
  DATA_TYPE *C = (DATA_TYPE*)malloc(4 * 4 * sizeof(DATA_TYPE));


  for (int i = 0; i < MAP_SIZE; ++i)
    for (int j = 0; j < MAP_SIZE; ++j)
      A[i * MAP_SIZE + j] = rand() / (float)RAND_MAX;

  // transformed filter
  WinogradConv2D_2x2_filter_transformation(C);

#ifdef USE_GPU
  gpu_selector dev_sel;
#else
  cpu_selector dev_sel;
#endif
  queue q(dev_sel);


  buffer<DATA_TYPE, 1> d_A (A, MAP_SIZE * MAP_SIZE);
  buffer<DATA_TYPE, 1> d_B ((MAP_SIZE-2) * (MAP_SIZE-2));
  buffer<DATA_TYPE, 1> d_C (C, 16);

  const int tile_n = (MAP_SIZE - 2 + 1) / 2;

  // initial problem size
  size_t globalWorkSize[2] = {
    (size_t)ceil(((float)tile_n) / ((float)DIM_LOCAL_WORK_GROUP_X)) * DIM_LOCAL_WORK_GROUP_X,
    (size_t)ceil(((float)tile_n) / ((float)DIM_LOCAL_WORK_GROUP_Y)) * DIM_LOCAL_WORK_GROUP_Y };

  size_t localWorkSize[2] = {DIM_LOCAL_WORK_GROUP_X, DIM_LOCAL_WORK_GROUP_Y};

  // adjust problem size for co-run
  size_t cpu_global_size[2];
  size_t gpu_global_size[2];
  size_t global_offset[2];

  bool pass = true;

  // sweep over cpu_offset 
  for (int cpu_offset = 0; cpu_offset <= 100; cpu_offset++) {

    double start = rtclock();

    cpu_global_size[0] = cpu_offset * (size_t)ceil(((float)tile_n) / ((float)DIM_LOCAL_WORK_GROUP_X)) 
      / 100 * DIM_LOCAL_WORK_GROUP_X;
    cpu_global_size[1] = globalWorkSize[1];
    gpu_global_size[0] = globalWorkSize[0] - cpu_global_size[0];
    gpu_global_size[1] = globalWorkSize[1];

    global_offset[0] = cpu_global_size[0];
    global_offset[1] = 0;

    range<2> gpu_gws(gpu_global_size[1], gpu_global_size[0]);
    range<2> cpu_gws(cpu_global_size[1], cpu_global_size[0]);
    range<2> lws(localWorkSize[1], localWorkSize[0]);
    id<2> gws_offset(global_offset[1], global_offset[0]);

    bool cpu_run = false, gpu_run = false;
    if (cpu_global_size[0] > 0) {
      cpu_run = true;
    }
    if (gpu_global_size[0] > 0) {
      gpu_run = true;
    }

    if (gpu_run) {
      q.submit([&] (handler &cgh) {
        auto input = d_A.get_access<sycl_read>(cgh);
        auto transformed_filter = d_C.get_access<sycl_read>(cgh);
        auto output = d_B.get_access<sycl_discard_write>(cgh);
        cgh.parallel_for<class winograd_conv2d>(nd_range<2>(gpu_gws, lws, gws_offset), [=] (nd_item<2> item) {

          int tile_j = item.get_global_id(0);
          int tile_i = item.get_global_id(1);

          // input transformation

          DATA_TYPE input_tile[4][4], tmp_tile[4][4], transformed_tile[4][4];
          for (int i = 0; i < 4; i ++) {
            for (int j = 0; j < 4; j ++) { 
              int x = 2 * tile_i + i;
              int y = 2 * tile_j + j;
              if (x >= MAP_SIZE || y >= MAP_SIZE) {
                input_tile[i][j] = 0;
                continue;
              }
              input_tile[i][j] = input[x * MAP_SIZE + y];
            }
          } 

          // Bt * d
          for (int j = 0; j < 4; j ++) {
            tmp_tile[0][j] = input_tile[0][j] - input_tile[2][j];
            tmp_tile[1][j] = input_tile[1][j] + input_tile[2][j];
            tmp_tile[2][j] = -input_tile[1][j] + input_tile[2][j];
            tmp_tile[3][j] = input_tile[1][j] - input_tile[3][j];
          }
          // d * B
          for (int i = 0; i < 4; i ++) {
            transformed_tile[i][0] = tmp_tile[i][0] - tmp_tile[i][2];
            transformed_tile[i][1] = tmp_tile[i][1] + tmp_tile[i][2];
            transformed_tile[i][2] = -tmp_tile[i][1] + tmp_tile[i][2];
            transformed_tile[i][3] = tmp_tile[i][1] - tmp_tile[i][3];
          }

          // element-wise multiplication

          DATA_TYPE multiplied_tile[4][4];
          for (int i = 0; i < 4; i ++) {
            for (int j = 0; j < 4; j ++) {
              multiplied_tile[i][j] = transformed_tile[i][j] * transformed_filter[i * 4 + j];
            }
          }

          // output transformation

          DATA_TYPE tmp_tile_1[2][4], final_tile[2][2];

          // At * I
          for (int j = 0; j < 4; j ++) {
            tmp_tile_1[0][j] = multiplied_tile[0][j] + multiplied_tile[1][j] + multiplied_tile[2][j];
            tmp_tile_1[1][j] = multiplied_tile[1][j] - multiplied_tile[2][j] - multiplied_tile[3][j];
          }
          // I * A
          for (int i = 0; i < 2; i ++) {
            final_tile[i][0] = tmp_tile_1[i][0] + tmp_tile_1[i][1] + tmp_tile_1[i][2];
            final_tile[i][1] = tmp_tile_1[i][1] - tmp_tile_1[i][2] - tmp_tile_1[i][3];
          }

          for (int i = 0; i < 2; i ++) {
            for (int j = 0; j < 2; j ++) {
              int x = 2 * tile_i + i;
              int y = 2 * tile_j + j;
              if (x >= MAP_SIZE - 2 || y >= MAP_SIZE - 2) {
                continue;
              }
              output[x * (MAP_SIZE - 2) + y] = final_tile[i][j];
            }
          }
        });
      });

    }

    if (cpu_run) {

      // printf("CPU size: %d\n", cpu_global_size[0]);
      WinogradConv2D_2x2_omp(A, B, C, cpu_global_size);

      q.submit([&] (handler &cgh) {
        auto acc = d_B.get_access<sycl_write>(cgh, gpu_run ? 
                                              global_offset[0]*2*(MAP_SIZE-2) : (MAP_SIZE-2)*(MAP_SIZE-2));
        cgh.copy(B, acc);
      });
    }

    q.submit([&] (handler &cgh) {
      auto acc = d_B.get_access<sycl_read>(cgh);
      cgh.copy(acc, B_outputFromGpu);
    }).wait();

    double end = rtclock();

#ifdef VERBOSE
    if (cpu_run) printf("run on host\n");
    if (gpu_run) printf("run on device\n");
    printf("CPU workload size : %d\n", cpu_offset);
#endif
    if (timing) printf("Total time: %lf ms\n", 1000.0 * (end - start));

    WinogradConv2D_2x2(A, B, C);
    pass &= compareResults(B, B_outputFromGpu);

  } // sweep

  printf("%s\n", pass ? "PASS" : "FAIL");

  free(A);
  free(B);
  free(B_outputFromGpu);
  free(C);
  return 0;
}
