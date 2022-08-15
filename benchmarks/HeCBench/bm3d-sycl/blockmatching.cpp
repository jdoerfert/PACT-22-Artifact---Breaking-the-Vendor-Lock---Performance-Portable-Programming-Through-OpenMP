#include "common.h"
#include "params.hpp"
#include "indices.hpp"


// Nearest lower power of 2
inline uint flp2 (uint x)
{
  return (0x80000000u >> sycl::clz(x));
}

//Computes the squared difference between two numbers
  template<typename T>
inline T L2p2(const T i1, const T i2)
{
  T diff = i1 - i2;
  return diff*diff;
}

/*
   Adds new patch to patch stack (only N most similar are kept)
Note: Stack is just an array, not FIFO
 */
void add_to_matched_image(
    uint *stack,         //IN/OUT: Stack of N patches matched to current reference patch
    uchar *num_patches_in_stack,//IN/OUT: Number of patches in stack
    const uint value,       //IN: [..DIFF(ushort)..|..LOC_Y(sbyte)..|..LOC_X(sbyte)..]
    const Params & params    //IN: Denoising parameters
    )
{
  //stack[*num_patches_in_stack-1] is most similar (lowest number)
  int k;

  uchar num = (*num_patches_in_stack);
  if (num < params.N) //add new value
  {
    k = num++;
    while(k > 0 && value > stack[k-1])
    {
      stack[k] = stack[k-1];
      --k;
    }

    stack[k] = value;
    *num_patches_in_stack = num;
  }
  else if (value >= stack[0]) 
    return;  
  else //delete highest value and add new
  {
    k = 1;
    while (k < params.N && value < stack[k])
    {
      stack[k-1] = stack[k];
      k++;
    }
    stack[k-1] = value;
  }
}

/*
   Block-matching algorithm
   For each processed reference patch it finds maximaly N similar patches that pass the distance threshold and stores them to the g_stacks array. 
   It also returns the number of them for each reference patch in g_num_patches_in_stack.
   Used denoising parameters: n,k,N,T,p
Division: Kernel handles gridDim.y lines starting with the line passed in argument. Each block handles warpSize reference patches in line. 
Each thread process one reference patch. All the warps of a block process the same reference patches.
 */
void block_matching(
    nd_item<2> item,
    uint* __restrict s_data,
    const  uchar* __restrict image, //IN: Original image
    ushort* __restrict g_stacks,         //OUT: For each reference patch contains addresses of similar patches (patch is adressed by top left corner) [..LOC_Y(sbyte)..|..LOC_X(sbyte)..]
    uint* __restrict g_num_patches_in_stack,  //OUT: For each reference patch contains number of similar patches
    const uint2 image_dim,      //IN: Image dimensions
    const uint2 stacks_dim,      //IN: Size of area, where reference patches could be located
    const Params params,      //IN: Denoising parameters
    const uint2 start_point)    //IN: Address of the top-left reference patch of a batch
{
  const int lidx = item.get_local_id(1);
  const int bidx = item.get_group(1);
  const int bidy = item.get_group(0);
  const int dimx = item.get_local_range(1);
  const int gridx = item.get_group_range(1);

  //One block is processing warpSize patches (because each warp is computing distance of same warpSize patches from different displaced patches)
  int tid = lidx % warpSize;
  int wid = lidx / warpSize;
  int num_warps = dimx/warpSize;

  //p_block denotes reference rectangle on which current cuda block is computing
  uint p_rectangle_width = ((warpSize-1) * params.p) + params.k;
  uint p_rectangle_start = start_point.x() + bidx * warpSize * params.p;

  //Shared arrays
  uint *s_diff = s_data; //SIZE: p_rectangle_width*num_warps
  uint *s_stacks = &s_data[p_rectangle_width*num_warps]; //SIZE: params.N*num_warps*warpSize
  uchar *s_patches_in_stack = (uchar*)&s_data[num_warps*(p_rectangle_width + params.N*warpSize)]; //SIZE: num_warps*warpSize
  uchar *s_image_p = (uchar*)&s_patches_in_stack[num_warps*warpSize]; //SIZE: p_rectangle_width*params.k

  s_diff += idx2(0, wid, p_rectangle_width);

  //Initialize s_patches_in_stack to zero
  s_patches_in_stack[ idx2(tid, wid, warpSize) ] = 0;

  int2 p; //Address of reference patch
  int2 q; //Address of patch against which the difference is computed

  p.x() = p_rectangle_start + (tid*params.p);
  p.y() = start_point.y() + (bidy*params.p);

  //Ensure, that the bottom most patches will be taken as reference patches regardless the p parameter.
  if (p.y() >= stacks_dim.y() && p.y() < stacks_dim.y() + params.p - 1)
    p.y() = stacks_dim.y() - 1;
  else if (p.y() >= stacks_dim.y()) return;

  //Ensure, that the right most patches will be taken as reference patches regardless the p parameter.
  uint inner_p_x = tid*params.p;
  if (p.x() >= stacks_dim.x() && p.x() < stacks_dim.x() + params.p - 1)
  {
    inner_p_x -= (p.x() - (stacks_dim.x() - 1));
    p.x() = stacks_dim.x() - 1;
  }

  //Load reference patches needed by actual block to shared memory
  for(int i = lidx; i < p_rectangle_width*params.k; i+=dimx)
  {
    int sx = i % p_rectangle_width;
    int sy = i / p_rectangle_width;
    if (p_rectangle_start+sx >= image_dim.x()) continue;
    s_image_p[i] = image[idx2(p_rectangle_start+sx,p.y()+sy,image_dim.x())];
  }

  item.barrier(access::fence_space::local_space);

  //scale difference so that it can fit ushort
  uint shift = (sycl::clz(params.Tn) < 16u) ? 16u - (uint)sycl::clz(params.Tn) : 0;


  //Ensure that displaced patch coordinates (q) will be positive
  int2 from;
  from.y() = (p.y() - (int)params.n < 0) ? -p.y() : -(int)params.n;
  from.x() = (((int)p_rectangle_start) - (int)params.n < 0) ? -((int)p_rectangle_start) : -(int)params.n;
  from.x() += wid;

  //For each displacement (x,y) in n neighbourhood
  for(int y = from.y(); y <= (int)params.n; ++y)
  {
    q.y() = p.y() + y;
    if (q.y() >= stacks_dim.y()) break;

    for(int x = from.x(); x <= (int)params.n; x += num_warps)
    {
      //Reference patch is always the most similar to itself (there is no need to copute it)
      if (x == 0 && y == 0) continue; 

      //Each warp is computing the same patch with slightly different displacement.
      //Compute distance of reference patch p from current patch q which is dispaced by (x+tid,y)

      //q_block denotes displaced rectangle which is processed by the current warp
      uint q_rectangle_start = p_rectangle_start + x;
      q.x() = q_rectangle_start + inner_p_x;

      //Compute distance for each column of reference patch
      for(uint i = tid; i < p_rectangle_width && p_rectangle_start+i < image_dim.x() && 
                        q_rectangle_start+i < image_dim.x(); i+=warpSize)
      {
        uint dist = 0;
        for(uint iy = 0; iy < params.k; ++iy)
        {
          dist += L2p2((int)s_image_p[ idx2(i, iy, p_rectangle_width) ], 
                       (int)image[ idx2(q_rectangle_start+i, q.y()+iy, image_dim.x()) ]);
        }
        s_diff[i] = dist;
      }

      if (p.x() >= stacks_dim.x() || q.x() >= stacks_dim.x()) continue;

      //Sum column distances to obtain patch distance
      uint diff = 0;
      for (uint i = 0; i < params.k; ++i) 
        diff += s_diff[inner_p_x + i];

      //Distance threshold
      if(diff < params.Tn)
      {
        uint loc_y = (uint)((q.y() - p.y()) & 0xFF); //relative location y (-127 to 127)
        uint loc_x = (uint)((q.x() - p.x()) & 0xFF); //relative location x (-127 to 127)
        diff >>= shift;
        diff <<= 16u; // [..DIFF(ushort)..|..LOC_Y(sbyte)..|..LOC_X(sbyte)..]
        diff |= (loc_y << 8u);
        diff |= loc_x;

        //Add current patch to s_stacks
        add_to_matched_image( 
            &s_stacks[ params.N * idx2(tid, wid, warpSize) ],
            &s_patches_in_stack[ idx2(tid, wid, warpSize) ],
            diff,
            params
            );
      }
    }
  }

  item.barrier(access::fence_space::local_space);

  uint batch_size = gridx*warpSize;
  uint block_address_x = bidx*warpSize+tid;

  if (wid > 0) return;
  //Select N most similar patches for each reference patch from stacks in shared memory and save them to global memory
  //Each thread represents one reference patch 
  //Each thread will find N most similar blocks in num_warps stacks (which were computed by different warps) and save them into global memory
  //In shared memory the most similar patch is at the end, in global memory the order does not matter
  //DEV: performance impact cca 8%
  if (p.x() >= stacks_dim.x()) return;

  int j;
  for (j = 0; j < params.N; ++j)
  {
    uint count = 0;
    uint minIdx = 0;
    uint minVal = 0xFFFFFFFF; //INF

    //Finds patch with minimal value of remaining
    for (int i = minIdx; i < num_warps; ++i)
    {
      count = (uint)s_patches_in_stack[ idx2(tid, i, warpSize) ];
      if (count == 0) continue;

      uint newMinVal = s_stacks[ idx3(count-1,tid,i,params.N,warpSize) ];
      if (newMinVal < minVal)
      {
        minVal = newMinVal;
        minIdx = i;
      }
    }
    if (minVal == 0xFFFFFFFF) break; //All stacks are empty

    //Remove patch from shared stack
    s_patches_in_stack[ idx2(tid, minIdx, warpSize) ]--;

    //Adds patch to stack in global memory
    g_stacks[idx3(j, block_address_x, bidy, params.N, batch_size)] = (ushort)(minVal & 0xFFFF);
  }
  //Save to the global memory the number of similar patches rounded to the nearest lower power of two
  g_num_patches_in_stack[ idx2(block_address_x ,bidy, batch_size) ] = flp2((uint)j+1)-1;
}


extern "C" void run_block_matching(
    queue &q,
    buffer<uchar, 1> &image, //Original image
    buffer<ushort, 1> &stacks, //For each reference patch contains addresses of similar patches (patch is adressed by top left corner)
    buffer<uint, 1> &num_patches_in_stack, //For each reference patch contains number of similar patches
    const uint2 image_dim,  //Image dimensions
    const uint2 stacks_dim, //size of area where reference patches could be located
    const Params params,    //Denoising parameters
    const uint2 start_point,//Address of the top-left reference patch of a batch
    const range<2> lws,  
    const range<2> gws,
    const uint shared_memory_size
    )
{
  q.submit([&] (handler &cgh) {
    auto image_acc = image.get_access<sycl_read>(cgh);
    auto stacks_acc = stacks.get_access<sycl_discard_write>(cgh);
    auto num_patches_in_stack_acc = num_patches_in_stack.get_access<sycl_discard_write>(cgh);
    accessor<uint, 1, sycl_read_write, access::target::local> 
      lmem(shared_memory_size/sizeof(uint), cgh);
    cgh.parallel_for<class bm>(nd_range<2>(gws, lws), [=] (nd_item<2> item) {
      block_matching(item,
                     lmem.get_pointer(),
                     image_acc.get_pointer(),
                     stacks_acc.get_pointer(),
                     num_patches_in_stack_acc.get_pointer(),
                     image_dim,
                     stacks_dim,
                     params,
                     start_point
                     );          
    });
  });
}
