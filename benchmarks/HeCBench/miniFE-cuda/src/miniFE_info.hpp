#ifndef miniFE_info_hpp
#define miniFE_info_hpp

#define MINIFE_HOSTNAME "lassen709"
#define MINIFE_KERNEL_NAME "'Linux'"
#define MINIFE_KERNEL_RELEASE "'4.14.0-115.21.2.1chaos.ch6a.ppc64le'"
#define MINIFE_PROCESSOR "'ppc64le'"

#define MINIFE_CXX "'/usr/tce/packages/cuda/cuda-11.1.0/bin/nvcc'"
#define MINIFE_CXX_VERSION "'nvcc: NVIDIA (R) Cuda compiler driver'"
#define MINIFE_CXXFLAGS "'-I. -I../utils -I../fem -DMINIFE_SCALAR=double -DMINIFE_LOCAL_ORDINAL=int -DMINIFE_GLOBAL_ORDINAL=int -DMINIFE_RESTRICT=__restrict__ -O3 -x cu -arch=sm_60 -DMINIFE_CSR_MATRIX '"

#endif
