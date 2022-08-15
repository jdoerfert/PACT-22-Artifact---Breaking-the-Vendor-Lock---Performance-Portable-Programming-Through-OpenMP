#pragma omp target teams distribute parallel for num_teams(globalpre/localpre) thread_limit(localpre)
for (int tid = 0; tid < DIVISIONS; tid++) {
  int sum = 0;
  for (int i = tid; i < size; i += DIVISIONS) {
    int x = d_prefixoffsets[i];
    d_prefixoffsets[i] = sum;
    sum += x;
  }
  h_offsets[tid] = sum;
}



