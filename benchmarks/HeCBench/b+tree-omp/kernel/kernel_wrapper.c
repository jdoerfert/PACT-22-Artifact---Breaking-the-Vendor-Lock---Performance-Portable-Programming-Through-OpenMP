#include <stdio.h>
#include <string.h>
#include <omp.h>
#include "../common.h"                // (in directory provided here)
#include "../util/timer/timer.h"          // (in directory provided here)
#include "./kernel_wrapper.h"      // (in directory provided here)


void 
kernel_wrapper(  record *records,
    long records_mem, // not length in byte
    knode *knodes,
    long knodes_elem,
    long knodes_mem,  // not length in byte

    int order,
    long maxheight,
    int count,

    long *currKnode,
    long *offset,
    int *keys,
    record *ans)
{

  //======================================================================================================================================================150
  //  CPU VARIABLES
  //======================================================================================================================================================150

  // timer
  long long offload_start = get_time();

  // findK kernel

  int threads = order < 256 ? order : 256;

#pragma omp target data map(to: knodes[0: knodes_mem],\
                                records[0: records_mem],\
                                keys[0: count], \
                                currKnode[0: count],\
                                offset[0: count])\
                        map(from: ans[0: count])
 {
#pragma omp target teams num_teams(count) thread_limit(threads)
    {
#pragma omp parallel
      {
        // private thread IDs
        int thid = omp_get_thread_num();
        int bid = omp_get_team_num();

        // processtree levels
        for(int i = 0; i < maxheight; i++){

          // if value is between the two keys
          if((knodes[currKnode[bid]].keys[thid]) <= keys[bid] && (knodes[currKnode[bid]].keys[thid+1] > keys[bid])){
            // this conditional statement is inserted to avoid crush due to but in original code
            // "offset[bid]" calculated below that addresses knodes[] in the next iteration goes outside of its bounds cause segmentation fault
            // more specifically, values saved into knodes->indices in the main function are out of bounds of knodes that they address
            if(knodes[offset[bid]].indices[thid] < knodes_elem){
              offset[bid] = knodes[offset[bid]].indices[thid];
            }
          }
#pragma omp barrier
          // set for next tree level
          if(thid==0){
            currKnode[bid] = offset[bid];
          }
#pragma omp barrier

        }

        //At this point, we have a candidate leaf node which may contain
        //the target record.  Check each key to hopefully find the record
        if(knodes[currKnode[bid]].keys[thid] == keys[bid]){
          ans[bid].value = records[knodes[currKnode[bid]].indices[thid]].value;
        }
      }
    }
  } 
  long long offload_end = get_time();

#ifdef DEBUG
  for (int i = 0; i < count; i++)
    printf("ans[%d] = %d\n", i, ans[i].value);
  printf("\n");
#endif

  printf("Device offloading time:\n");
  printf("%.12f s\n", (float) (offload_end-offload_start) / 1000000);

}

