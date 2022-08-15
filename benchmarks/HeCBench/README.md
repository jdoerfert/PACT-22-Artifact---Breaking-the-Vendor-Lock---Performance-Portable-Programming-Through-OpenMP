# HeCBench
This repository contains a collection of Heterogeneous Computing benchmarks written with CUDA, HIP, SYCL (DPC++), and OpenMP-4.5 target offloading for studying performance, portability, and productivity. 

# Software installation
[AMD ROCm](https://rocmdocs.amd.com/en/latest/Installation_Guide/Installation-Guide.html)  
[Intel DPC++ compiler](https://github.com/intel/llvm/blob/sycl/sycl/doc/GetStartedGuide.md) or [Intel oneAPI toolkit](https://software.intel.com/content/www/us/en/develop/articles/installation-guide-for-intel-oneapi-toolkits.html)  
[Nvidia HPC SDK](https://developer.nvidia.com/hpc-sdk)

# Dataset
For Rodinia benchmarks, please download the dataset at http://lava.cs.virginia.edu/Rodinia/download.htm 

# Known issues
The programs have not been evaluated on Windows or MacOS  
The lastest Intel SYCL compiler (not the Intel oneAPI toolkit) may be needed for building some SYCL programs successfully  
Kernel results do not exactly match using these programming languages on a platform for certain programs  
Not all programs automate the verification of host and device results  
Not all CUDA programs have SYCL, HIP or OpenMP equivalents  
Not all programs have OpenMP target offloading implementations  
Raw performance of any program may be suboptimal  
Some programs may take longer to complete on an integrated GPU  
Some host programs contain platform-specific intrinsics, so they may cause compile error on a PowerPC platform

# Feedback
I appreciate your feedback when any examples don't look right.

# Experimental Results
Early results are shown [here](results/README.md)

# Reference
### ace (cuda)
  Phase-field simulation of dendritic solidification (https://github.com/myousefi2016/Allen-Cahn-CUDA)

### adv (cuda)
  Advection (https://github.com/Nek5000/nekBench/tree/master/adv)

### aes (opencl)
  AES encrypt and decrypt (https://github.com/Multi2Sim/m2s-bench-amdsdk-2.5-src)

### affine (opencl)
  Affine transformation (https://github.com/Xilinx/SDAccel_Examples/tree/master/vision/affine)

### aidw (cuda)
  Adaptive inverse distance weighting (Mei, G., Xu, N. & Xu, L. Improving GPU-accelerated adaptive IDW interpolation algorithm using fast kNN search. SpringerPlus 5, 1389 (2016))

### aligned-types (cuda)
  Alignment specification for variables of structured types (http://docs.nvidia.com/cuda/cuda-samples/index.html)

### all-pairs-distance (cuda)
  All-pairs distance calculation (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2910913/)

### amgmk (openmp)
  The relax kernel in the AMGmk benchmark (https://asc.llnl.gov/CORAL-benchmarks/Micro/amgmk-v1.0.tar.gz)
  
### ans (cuda)
  Asymmetric numeral systems decoding (https://github.com/weissenberger/multians)
  
### aobench (openmp)
  A lightweight ambient occlusion renderer (https://code.google.com/archive/p/aobench)

### aop (cuda)
  American options pricing (https://github.com/NVIDIA-developer-blog)

### asmooth (cuda)
  Adaptive smoothing (http://www.hcs.harvard.edu/admiralty/)

### asta (cuda)
  Array of structure of tiled array for data layout transposition (https://github.com/chai-benchmarks/chai)

### atomicIntrinsics (cuda)
  Atomic add, subtract, min, max, AND, OR, XOR (http://docs.nvidia.com/cuda/cuda-samples/index.html)

### atomicCAS (cuda)
  64-bit atomic add, min, and max with compare and swap (https://github.com/treecode/Bonsai/blob/master/runtime/profiling/derived_atomic_functions.h)

### atomicReduction (hip)
  Integer sum reduction with atomics (https://github.com/ROCm-Developer-Tools/HIP-Examples/tree/master/reduction)

### attention (pseudocodes)
  Ham, T.J., et al., 2020, February. A^ 3: Accelerating Attention Mechanisms in Neural Networks with Approximation. In 2020 IEEE International Symposium on High Performance Computer Architecture (HPCA) (pp. 328-341). IEEE.

### axhelm (cuda)
  Helmholtz matrix-vector product (https://github.com/Nek5000/nekBench/tree/master/axhelm)

### babelstream (cuda)
  Measure memory transfer rates for copy, add, mul, triad, dot, and nstream (https://github.com/UoB-HPC/BabelStream)

### backprop (opencl)
  Backpropagation in the Rodinia benchmark suite (http://lava.cs.virginia.edu/Rodinia/download_links.htm)

### bezier-surface (opencl)
  The Bezier surface (https://github.com/chai-benchmarks/chai)

### bfs (opencl)
  The breadth-first search in the Rodinia benchmark suite (http://lava.cs.virginia.edu/Rodinia/download_links.htm)

### bh (cuda)
  Simulate the gravitational forces in a star cluster using the Barnes-Hut n-body algorithm (https://userweb.cs.txstate.edu/~burtscher/research/ECL-BH/)

### bilateral (cuda)
  Bilateral filter (https://github.com/jstraub/cudaPcl)

### binomial (cuda)
  Evaluate fair call price for a given set of European options under binomial model (https://docs.nvidia.com/cuda/cuda-samples/index.html)

### bitonic-sort (sycl)
  Bitonic sorting (https://github.com/oneapi-src/oneAPI-samples/tree/master/DirectProgramming/)

### bitpacking (cuda)
  A bit-level operation that aims to reduce the number of bits required to store each value (https://github.com/NVIDIA/nvcomp)

### black-scholes (cuda)
  The Black-Scholes simulation (https://github.com/cavazos-lab/FinanceBench)

### bm3d (cuda)
  Block-matching and 3D filtering method for image denoising (https://github.com/DawyD/bm3d-gpu)

### bn (cuda)
  Bayesian network learning (https://github.com/OSU-STARLAB/UVM_benchmark/blob/master/non_UVM_benchmarks)

### bonds (cuda)
  Fixed-rate bond with flat forward curve (https://github.com/cavazos-lab/FinanceBench)

### boxfilter (cuda)
  Box filtering (http://developer.download.nvidia.com/compute/cuda/3_0/sdk/website/OpenCL/website/samples.html)

### bsearch (cuda)
  Classic and vectorizable binary search algorithms (https://www.sciencedirect.com/science/article/abs/pii/S0743731517302836)

### bspline-vgh (openmp)
  Bspline value gradient hessian (https://github.com/QMCPACK/miniqmc/blob/OMP_offload/src/OpenMP/main.cpp)

### bsw (cuda)
  GPU accelerated Smith-Waterman for performing batch alignments (https://github.com/mgawan/ADEPT)

### burger (openmp)
  2D Burger's equation (https://github.com/soumyasen1809/OpenMP_C_12_steps_to_Navier_Stokes)

### bwt (cuda)
  Burrows-Wheeler transform (https://github.com/jedbrooke/cuda_bwt)

### b+tree (opencl)
  B+Tree in the Rodinia benchmark suite (http://lava.cs.virginia.edu/Rodinia/download_links.htm)

### cbsfil (cuda)
  Cubic b-spline filtering (https://github.com/DannyRuijters/CubicInterpolationCUDA)

### ccs (cuda)
  Condition-dependent Correlation Subgroups (https://github.com/abhatta3/Condition-dependent-Correlation-Subgroups-CCS)

### ccsd-trpdrv (c)
  The CCSD tengy kernel, which was converted from Fortran to C by Jeff Hammond, in NWChem (https://github.com/jeffhammond/nwchem-ccsd-trpdrv)

### ced (opencl)
  Canny edge detection (https://github.com/chai-benchmarks/chai)

### cfd (opencl)
  The CFD solver in the Rodinia benchmark suite (http://lava.cs.virginia.edu/Rodinia/download_links.htm)

### che (cuda)
  Phase-field simulation of spinodal decomposition using the Cahn-Hilliard equation (https://github.com/myousefi2016/Cahn-Hilliard-CUDA)

### chemv (cuda)
  Complex hermitian matrix-vector multiplication (https://repo.or.cz/ppcg.git)

### chi2 (cuda)
  The Chi-square 2-df test. (https://web.njit.edu/~usman/courses/cs677_spring19/)

### clenergy (opencl)
  Direct coulomb summation kernel (http://www.ks.uiuc.edu/Training/Workshop/GPU_Aug2010/resources/clenergy.tar.gz)

### clink (c)
  Compact LSTM inference kernel (http://github.com/UCLA-VAST/CLINK)

### cmp (cuda)
  Seismic processing using the classic common midpoint (CMP) method (https://github.com/hpg-cepetro/IPDPS-CRS-CMP-code)

### cobahh (opencl)
  Simulation of Random Network of Hodgkin and Huxley Neurons with Exponential Synaptic Conductances (https://dl.acm.org/doi/10.1145/3307339.3343460)

### columnarSolver (cuda)
  Dimitrov, M. and Esslinger, B., 2021. CUDA Tutorial--Cryptanalysis of Classical Ciphers Using Modern GPUs and CUDA. arXiv preprint arXiv:2103.13937.

### compute-score (opencl)
  Document filtering (https://www.intel.com/content/www/us/en/programmable/support/support-resources/design-examples/design-software/opencl/compute-score.html)

### convolutionSeperable (opencl)
  Convolution filter of a 2D image with separable kernels (http://developer.download.nvidia.com/compute/cuda/3_0/sdk/website/OpenCL/website/samples.html)

### crc64 (openmp)
  64-bit cyclic-redundancy check (https://xgitlab.cels.anl.gov/hfinkel/hpcrc64/-/wikis/home)

### crs (cuda)
  Cauchy Reed-Solomon encoding (https://www.comp.hkbu.edu.hk/~chxw/gcrs.html)

### d2q9_bgk (sycl)
  A lattice boltzmann scheme with a 2D grid, 9 velocities, and Bhatnagar-Gross-Krook collision step (https://github.com/WSJHawkins/ExploringSycl)

### dct8x8 (opencl)
  Discrete Cosine Transform (DCT) and inverse DCT for 8x8 blocks (http://developer.download.nvidia.com/compute/cuda/3_0/sdk/website/OpenCL/website/samples.html)

### ddbp (cuda)
  Distance-driven backprojection (https://github.com/LAVI-USP/DBT-Reconstruction)

### debayer (opencl)
  Convert a Bayer mosaic raw image to RGB (https://github.com/GrokImageCompression/latke)

### degrid (cuda)
  Radio astronomy degridding (https://github.com/NVIDIA/SKA-gpu-degrid)
  
### deredundancy (sycl)
  Gene sequence de-redundancy is a precise gene sequence de-redundancy software that supports heterogeneous acceleration (https://github.com/JuZhenCS/gene-sequences-de-redundancy)
  
### diamond (opencl)
  Mask sequences kernel in Diamond (https://github.com/bbuchfink/diamond)

### divergence (cuda)
  CPU and GPU divergence test (https://github.com/E3SM-Project/divergence_cmdvse)

### dp (opencl)
  Dot product (http://developer.download.nvidia.com/compute/cuda/3_0/sdk/website/OpenCL/website/samples.html)

### dslash (sycl)
  A Lattice QCD Dslash operator proxy application derived from MILC (https://gitlab.com/NERSC/nersc-proxies/milc-dslash)

### dxtc1 (opencl)
  DXT1 compression (http://developer.download.nvidia.com/compute/cuda/3_0/sdk/website/OpenCL/website/samples.html)

### easyWave (cuda)
  Simulation of tsunami generation and propagation in the context of early warning (https://gitext.gfz-potsdam.de/geoperil/easyWave)

### eigenvalue (opencl)
  Calculate the eigenvalues of a tridiagonal symmetric matrix (https://github.com/OpenCL/AMD_APP_samples)

### entropy (cuda)
  Compute the entropy for each point of a 2D matrix using a 5x5 window (https://lan-jing.github.io/parallel%20computing/system/entropy/)

### epistatis (sycl)
  Epistasis detection (https://github.com/rafatcampos/bio-epistasis-detection)
   
### ert (cuda)
  Modified microkernel in the empirical roofline tool (https://bitbucket.org/berkeleylab/cs-roofline-toolkit/src/master/)
   
### extend2 (c)
  Smith-Waterman (SW) extension in Burrow-wheeler aligner for short-read alignment (https://github.com/lh3/bwa)

### extrema (cuda)
  Find local maxima (https://github.com/rapidsai/cusignal/)

### f16max (c)
  Compute the maximum of half-precision floating-point numbers using bit operations (https://x.momo86.net/en?p=113)

### f16sp (cuda)
  Half-precision scalar product (https://docs.nvidia.com/cuda/cuda-samples/index.html)

### face (cuda)
  Face detection using the Viola-Jones algorithm (https://sites.google.com/site/5kk73gpu2012/assignment/viola-jones-face-detection)

### fdtd3d (opencl)
  FDTD-3D (http://developer.download.nvidia.com/compute/cuda/3_0/sdk/website/OpenCL/website/samples.html)

### feynman-kac (c)
  Use of Feynman-Kac algorithm to solve Poisson's equation in a 2D ellipse (https://people.sc.fsu.edu/~jburkardt/c_src/feynman_kac_2d/feynman_kac_2d.html)

### fhd (cuda)
  A case study: advanced magnetic resonance imaging reconstruction (https://ict.senecacollege.ca/~gpu610/pages/content/cudas.html)
  
### filter (cuda)
  Filtering by a predicate (https://developer.nvidia.com/blog/cuda-pro-tip-optimized-filtering-warp-aggregated-atomics/)

### fft (opencl)
  FFT in the SHOC benchmark suite(https://github.com/vetter/shoc/)

### flame (cuda)
  Fractal flame (http://gpugems.hwu-server2.crhc.illinois.edu/)

### floydwarshall (hip)
  Floyd-Warshall Pathfinding sample (https://github.com/ROCm-Developer-Tools/HIP-Examples/blob/master/HIP-Examples-Applications/FloydWarshall/)

### fluidSim (opencl)
  2D Fluid Simulation using the Lattice-Boltzman method (https://github.com/OpenCL/AMD_APP_samples)

### fpc (opencl)
  Frequent pattern compression ( Base-delta-immediate compression: practical data compression for on-chip caches. In Proceedings of the 21st international conference on Parallel architectures and compilation techniques (pp. 377-
388). ACM.)

### fresnel (c)
  Fresnel integral (http://www.mymathlib.com/functions/fresnel_sin_cos_integrals.html)

### frna (cuda)
  Accelerate the fill step in predicting the lowest free energy structure and a set of suboptimal structures (http://rna.urmc.rochester.edu/Text/Fold-cuda.html)

### fwt (cuda)
  Fast Walsh transformation (http://docs.nvidia.com/cuda/cuda-samples/index.html)

### ga (cuda)
  Gene alignment (https://github.com/NUCAR-DEV/Hetero-Mark)

### gamma-correction (sycl)
  Gamma correction (https://github.com/intel/BaseKit-code-samples)

### gaussian (opencl)
  Gaussian elimination in the Rodinia benchmark suite (http://lava.cs.virginia.edu/Rodinia/download_links.htm)

### gc (cuda)
  Graph coloring via shortcutting (https://userweb.cs.txstate.edu/~burtscher/research/ECL-GC/)

### gd (c++)
  Gradient descent (https://github.com/CGudapati/BinaryClassification)

### geodesic (opencl)
  Geodesic distance (https://www.osti.gov/servlets/purl/1576565)

### gmm (cuda)
  Expectation maximization with Gaussian mixture models (https://github.com/Corv/CUDA-GMM-MultiGPU)

### goulash (cuda)
  Simulate the dynamics of a small part of a cardiac myocyte, specifically the fast sodium m-gate  (https://github.com/LLNL/goulash)
### gpp (cuda, omp)
  General Plasman Pole Self-Energy Simulation the BerkeleyGW software package (https://github.com/NERSC/gpu-for-science-day-july-2019)

### grep (cuda)
  Regular expression matching (https://github.com/bkase/CUDA-grep)

### grrt (cuda)
  General-relativistic radiative transfer calculations coupled with the calculation of geodesics in the Kerr spacetime (https://github.com/hungyipu/Odyssey)

### haccmk (c)
  The HACC microkernel (https://asc.llnl.gov/CORAL-benchmarks/#haccmk)

### heartwall (opencl)
  Heart Wall in the Rodinia benchmark suite (http://lava.cs.virginia.edu/Rodinia/download_links.htm)

### heat (sycl)
  The heat equation solver (https://github.com/UoB-HPC/heat_sycl)

### heat2d (cuda)
  Discrete 2D laplacian operation a number of times on a given vector (https://github.com/gpucw/cuda-lapl)
  
### hellinger (cuda)
  Hellinger distance (https://github.com/rapidsai/raft)

### henry (cuda)
  Henry coefficient (https://github.com/CorySimon/HenryCoefficient)

### hexicton (opencl)
  A Portable and Scalable Solver-Framework for the Hierarchical Equations of Motion (https://github.com/noma/hexciton_benchmark)
  
### histogram (cuda)
  Histogram (http://github.com/NVlabs/cub/tree/master/experimental)

### hmm (opencl)
  Hidden markov model (http://developer.download.nvidia.com/compute/DevZone/OpenCL/Projects/oclHiddenMarkovModel.tar.gz)

### hogbom (cuda)
  The benchmark implements the kernel of the Hogbom Clean deconvolution algorithm (https://github.com/ATNF/askap-benchmarks/)

### hotspot3D (opencl)
  Hotspot3D in the Rodinia benchmark suite (http://lava.cs.virginia.edu/Rodinia/download_links.htm)

### hwt1d (opencl)
  1D Haar wavelet transformation (https://github.com/OpenCL/AMD_APP_samples)

### hybridsort (opencl)
  Hybridsort in the Rodinia benchmark suite (http://lava.cs.virginia.edu/Rodinia/download_links.htm)

### idivide (cuda)
  Fast interger divide (https://github.com/milakov/int_fastdiv)

### interleave (cuda)
  Interleaved and non-interleaved global memory accesses (Shane Cook. 2012. CUDA Programming: A Developer's Guide to Parallel Computing with GPUs (1st. ed.). Morgan Kaufmann Publishers Inc., San Francisco, CA, USA.)

### inversek2j (cuda)
  The inverse kinematics for 2-joint arm (http://axbench.org/)

### is (cuda)
  Integer sort (https://github.com/GMAP/NPB-GPU)

### ising (cuda)
  Monte-Carlo simulations of 2D Ising Model (https://github.com/NVIDIA/ising-gpu/)

### iso2dfd (sycl)
  Isotropic 2-dimensional Finite Difference (https://github.com/intel/HPCKit-code-samples/)

### jaccard (cuda)
  Jaccard index for a sparse matrix (https://github.com/rapidsai/nvgraph/blob/main/cpp/src/jaccard_gpu.cu)

### jacobi (cuda)
  Jacobi relaxation (https://github.com/NVIDIA/multi-gpu-programming-models/blob/master/single_gpu/jacobi.cu)

### jenkins-hash (c)
  Bob Jenkins lookup3 hash function (https://android.googlesource.com/platform/external/jenkins-hash/+/75dbeadebd95869dd623a29b720678c5c5c55630/lookup3.c)

### kalman (cuda)
  Kalman filter (https://github.com/rapidsai/cuml/)  

### keccaktreehash (cuda)
  A Keccak tree hash function (http://sites.google.com/site/keccaktreegpu/)

### keogh (cuda)
  Keogh's lower bound (https://github.com/gravitino/cudadtw)

### kmeans (opencl)
  K-means in the Rodinia benchmark suite (http://lava.cs.virginia.edu/Rodinia/download_links.htm)

### knn (cuda)
  K-nearest neighbor (https://github.com/OSU-STARLAB/UVM_benchmark/blob/master/non_UVM_benchmarks)

### lanczos (cuda)
  Lanczos tridiagonalization (https://github.com/linhr/15618)

### langford (cuda)
  Count planar Langford sequences (https://github.com/boris-dimitrov/z4_planar_langford_multigpu)

### laplace (cuda)
  A Laplace solver using red-black Gaussian Seidel with SOR solver (https://github.com/kyleniemeyer/laplace_gpu)

### lavaMD (opencl)
  LavaMD in the Rodinia benchmark suite (http://lava.cs.virginia.edu/Rodinia/download_links.htm)

### layout (opencl)
  AoS and SoA comparison (https://github.com/OpenCL/AMD_APP_samples)

### lda (cuda)
  Latent Dirichlet allocation (https://github.com/js1010/cusim)

### ldpc (cuda)
  QC-LDPC decoding (https://github.com/robertwgh/cuLDPC)

### lebesgue (c)
  Estimate the Lebesgue constant (https://people.math.sc.edu/Burkardt/c_src/lebesgue/lebesgue.html)

### leukocyte  (opencl)
  Leukocyte in the Rodinia benchmark suite (http://lava.cs.virginia.edu/Rodinia/download_links.htm)

### libor (cuda)
  A LIBOR market model Monte Carlo application (https://people.maths.ox.ac.uk/~gilesm/cuda_old.html)

### lid-driven-cavity  (cuda)
  GPU solver for a 2D lid-driven cavity problem (https://github.com/kyleniemeyer/lid-driven-cavity_gpu)

### lif (cuda)
   A leaky integrate-and-fire neuron model (https://github.com/e2crawfo/hrr-scaling)

### linearprobing (cuda)
  A simple lock-free hash table (https://github.com/nosferalatu/SimpleGPUHashTable)

### lombscargle (cuda)
  Lomb-Scargle periodogram (https://github.com/rapidsai/cusignal/)

### loopback (cuda)
  Lookback option simulation (https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-37-efficient-random-number-generation-and-application)

### lsqt (cuda)
  Linear scaling quantum transport (https://github.com/brucefan1983/gpuqt)

### lud (opencl)
  LU decomposition in the Rodinia benchmark suite (http://lava.cs.virginia.edu/Rodinia/download_links.htm)

### lulesh (cuda)
  Livermore unstructured Lagrangian explicit shock hydrodynamics (https://github.com/LLNL/LULESH)

### mandelbrot (sycl)
  The Mandelbrot set in the HPCKit code samples (https://github.com/intel/HPCKit-code-samples/)

### marchingCubes (cuda)
  A practical isosurfacing algorithm for large data on many-core architectures (https://github.com/LRLVEC/MarchingCubes)

### match (cuda)
  Compute matching scores for two 16K 128D feature points (https://github.com/Celebrandil/CudaSift)

### matrix-rotate (openmp)
  In-place matrix rotation

### maxpool3d (opencl)
  3D Maxpooling (https://github.com/nachiket/papaa-opencl)

### maxFlops (opencl)
  Maximum floating-point operations in the SHOC benchmark suite (https://github.com/vetter/shoc/)

### mcmd (cuda)
  Monte Carlo and Molecular Dynamics Simulation Package (https://github.com/khavernathy/mcmd)

### md (opencl)
  Molecular dynamics function in the SHOC benchmark suite (https://github.com/vetter/shoc/)

### mdh (opencl)
  Simple multiple Debye-Huckel kernel in fast molecular electrostatics algorithms on GPUs (http://gpugems.hwu-server2.crhc.illinois.edu/)
  
### md5hash (opencl)
  MD5 hash function in the SHOC benchmark suite (https://github.com/vetter/shoc/)

### medianfilter (opencl)
  Two-dimensional 3x3 median filter of RGBA image (http://developer.download.nvidia.com/compute/cuda/3_0/sdk/website/OpenCL/website/samples.html)
  
### memcpy (cuda)
  A benchmark for memory copy from a host to a device

### memtest (cuda)
  Selected memory tests (https://github.com/ComputationalRadiationPhysics/cuda_memtest)

### merge (cuda)
  Merge two unsorted arrays into a sorted array (https://github.com/ogreen/MergePathGPU)

### metropolis (cuda)
   Simulation of an ensemble of replicas with Metropolis–Hastings computation in the trial run (https://github.com/crinavar/trueke) 

### miniFE (omp)
  MiniFE Mantevo mini-application (https://github.com/Mantevo/miniFE)

### minibude (sycl)
  The core computation of the Bristol University Docking Engine (BUDE) (https://github.com/UoB-HPC/miniBUDE)

### minimap2 (cuda)
  Hardware acceleration of long read pairwise overlapping in genome sequencing (https://github.com/UCLA-VAST/minimap2-acceleration)

### minimod (cuda)
  A finite difference solver for seismic modeling (https://github.com/rsrice/gpa-minimod-artifacts)

### minisweep (openmp)
  A deterministic Sn radiation transport miniapp (https://github.com/wdj/minisweep)

### minkowski (cuda)
  Minkowski distance (https://github.com/rapidsai/raft)

### mis (cuda)
  Maximal independent set (http://www.cs.txstate.edu/~burtscher/research/ECL-MIS/)

### mixbench (cuda)
  A read-only version of mixbench (https://github.com/ekondis/mixbench)

### mkl-sgemm (sycl) 
  Single-precision floating-point matrix multiply using Intel<sup>®</sup> Math Kernel Library 

### mmcsf (cuda)
  MTTKRP kernel using mixed-mode CSF (https://github.com/isratnisa/MM-CSF)

### mnist (cuda)
  Chapter 4.2: Converting CUDA CNN to HIP (https://developer.amd.com/wp-content/resources)

### morphology (cuda)
  Morphological operators: Erosion and Dilation (https://github.com/yszheda/CUDA-Morphology)

### mr (c)
  The Miller-Rabin primality test (https://github.com/wizykowski/miller-rabin)

### mt (opencl)
  Mersenne Twister (http://developer.download.nvidia.com/compute/cuda/3_0/sdk/website/OpenCL/website/samples.html)

### multimaterial (sycl)
  Multi-material simulations (https://github.com/reguly/multimaterial)

### murmurhash3 (c)
  MurmurHash3 yields a 128-bit hash value (https://github.com/aappleby/smhasher/wiki/MurmurHash3)

### myocte (opencl)
  Myocte in the Rodinia benchmark suite (http://lava.cs.virginia.edu/Rodinia/download_links.htm)

## nbnxm (sycl)
  Computing non-bonded pair interactions (https://manual.gromacs.org/current/doxygen/html-full/page_nbnxm.xhtml)

## nbody (opencl)
  Nbody simulation (https://github.com/oneapi-src/oneAPI-samples/tree/master/DirectProgramming/DPC%2B%2B/N-BodyMethods/Nbody)

### ne (cuda)
  Normal estimation in 3D (https://github.com/PointCloudLibrary/pcl)

### nms (cuda)
  Work-efficient parallel non-maximum suppression kernels (https://github.com/hertasecurity/gpu-nms)

### nn (opencl)
  Nearest neighbor in the Rodinia benchmark suite (http://lava.cs.virginia.edu/Rodinia/download_links.htm)

### nqueen (cuda)
  N-Queens (https://github.com/tcarneirop/ChOp)

### ntt (cuda)
  Number-theoretic transform (https://github.com/vernamlab/cuHE)

### nw (opencl)
  Needleman-Wunsch in the Rodinia benchmark suite (http://lava.cs.virginia.edu/Rodinia/download_links.htm)

### openmp (cuda)
  Multi-threading over a single device (https://docs.nvidia.com/cuda/cuda-samples/index.html)

### page-rank (opencl)
  PageRank (https://github.com/Sable/Ostrich/tree/master/map-reduce/page-rank)

### particle-diffusion (sycl)
  Particle diffusion in the HPCKit code samples (https://github.com/intel/HPCKit-code-samples/)

### particlefilter (opencl)
  Particle Filter in the Rodinia benchmark suite (http://lava.cs.virginia.edu/Rodinia/download_links.htm)

### particles (opencl)
  Particles collision simulation (http://developer.download.nvidia.com/compute/cuda/3_0/sdk/website/OpenCL/website/samples.html)

### pathfinder (opencl)
  PathFinder in the Rodinia benchmark suite (http://lava.cs.virginia.edu/Rodinia/download_links.htm)

### perplexity (cuda)
  Perplexity search (https://github.com/rapidsai/cuml/)  

### phmm (cuda)
  Pair hidden Markov model (https://github.com/lienliang/Pair_HMM_forward_GPU)

### pointwise (cuda)
  Fused point-wise operations (https://developer.nvidia.com/blog/optimizing-recurrent-neural-networks-cudnn-5/)

### pool (hip)
  Pooling layer (https://github.com/PaddlePaddle/Paddle)

### popcount (opencl)
  Implementations of population count (Jin, Z. and Finkel, H., 2020, May. Population Count on Intel® CPU, GPU and FPGA. In 2020 IEEE International Parallel and Distributed Processing Symposium Workshops (IPDPSW) (pp. 432-439). IEEE.)

### present (c)
  Lightweight cryptography (https://github.com/bozhu/PRESENT-C/blob/master/present.h)

### prna (cuda)
  Calculate a partition function for a sequence, which can be used to predict base pair probabilities (http://rna.urmc.rochester.edu/Text/partition-cuda.html)

### projectile (sycl)
  Projectile motion is a program that implements a ballistic equation (https://github.com/intel/BaseKit-code-samples)

### qrg (cuda)
  Niederreiter quasirandom number generator and Moro's Inverse Cumulative Normal Distribution generator (http://developer.download.nvidia.com/compute/cuda/3_0/sdk/website/OpenCL/website/samples.html)

### qtclustering (opencl)
  quality threshold clustering (https://github.com/vetter/shoc/)

### quicksort (sycl)
  Quicksort (https://software.intel.com/content/www/us/en/develop/download/code-for-the-parallel-universe-article-gpu-quicksort-from-opencl-to-data-parallel-c.html)

### radixsort (opencl)
  A parallel radix sort (http://developer.download.nvidia.com/compute/cuda/3_0/sdk/website/OpenCL/website/samples.html)
  
### randomAccess (openmp)
  Random memory access (https://icl.cs.utk.edu/projectsfiles/hpcc/RandomAccess/)
  
### reaction (cuda)
  3D Gray-Scott reaction diffusion (https://github.com/ifilot/wavefuse)

### recursiveGaussian (opencl)
  2-dimensional Gaussian Blur Filter of RGBA image (http://developer.download.nvidia.com/compute/cuda/3_0/sdk/website/OpenCL/website/samples.html)

### reverse (cuda)
  Reverse an input array of size 256 using shared memory

### rfs (cuda)
  Reproducible floating sum (https://github.com/facebookarchive/fbcuda)

### rng-wallace (cuda)
  Random number generation using the Wallace algorithm (https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-37-efficient-random-number-generation-and-application)

### romberg (cuda)
  Romberg's method (https://github.com/SwayambhuNathRay/Parallel-Romberg-Integration)

### rsbench (opencl)
  A proxy application for full neutron transport application like OpenMC that support multipole cross section representations
  (https://github.com/ANL-CESAR/RSBench/)

### rtm8 (hip)
  A structured-grid applications in the oil and gas industry (https://github.com/ROCm-Developer-Tools/HIP-Examples/tree/master/rtm8)

### rushlarsen (cuda)
  An ODE solver using the Rush-Larsen scheme (https://bitbucket.org/finsberg/gotran/src/master)

### s3d (opencl)
  Chemical rates computation used in the simulation of combustion (https://github.com/vetter/shoc/)

### sad (cuda)
  Naive template matching with SAD (https://github.com/gholomia/CTMC)

### sampling (cuda)
  Shapley sampling values explanation method (https://github.com/rapidsai/cuml)

### scan (cuda)
  A block-level scan (https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda)

### scan2 (opencl)
  Scan a large array (https://github.com/OpenCL/AMD_APP_samples)

### secp256k1 (cuda)
  Part of BIP39 solver (https://github.com/johncantrell97/bip39-solver-gpu)

### sheath (cuda)
  Plasma sheath simulation with the particle-in-cell method (https://www.particleincell.com/2016/cuda-pic/)

### shmembench (cuda)
  The shared local memory microbenchmark (https://github.com/ekondis/gpumembench)

### shuffle (hip)
  Shuffle instructions with subgroup sizes of 8, 16, and 32 (https://github.com/cpc/hipcl/tree/master/samples/4_shfl)

### simplemoc (opencl)
  The attentuation of neutron fluxes across an individual geometrical segment (https://github.com/ANL-CESAR/SimpleMOC-kernel)

### slu (cuda)
  Sparse LU factorization (https://github.com/sheldonucr/GLU_public)

### snake (cuda)
  Genome pre-alignment filtering (https://github.com/CMU-SAFARI/SneakySnake)

### sobel (opencl)
  Sobel filter (https://github.com/OpenCL/AMD_APP_samples)

### sobol (cuda)
  Sobol quasi-random generator (https://docs.nvidia.com/cuda/cuda-samples/index.html)

### softmax (opencl)
  The softmax function (https://github.com/pytorch/glow/tree/master/lib/Backends/OpenCL)

### sort (opencl)
  Radix sort in the SHOC benchmark suite(https://github.com/vetter/shoc/)

### sosfil (cuda)
  Second-order IIR digital filtering (https://github.com/rapidsai/cusignal/)

### sparkler (cuda)
  A miniapp for the CoMet comparative genomics application (https://github.com/wdj/sparkler)

### sph (openmp)
  The simple n^2 SPH simulation (https://github.com/olcf/SPH_Simple)

### split (cuda)
  The split operation in radix sort (http://developer.download.nvidia.com/compute/cuda/3_0/sdk/website/OpenCL/website/samples.html)

### spm (cuda)
  Image registration calculations for the statistical parametric mapping (SPM) system (http://mri.ee.ntust.edu.tw/cuda/)

### sptrsv (cuda)
  A thread-Level synchronization-free sparse triangular solver (https://github.com/JiyaSu/CapelliniSpTRSV)

### srad (opencl)
  SRAD (version 1) in the Rodinia benchmark suite (http://lava.cs.virginia.edu/Rodinia/download_links.htm)

### ss (opencl)
  String search (https://github.com/OpenCL/AMD_APP_samples)

### sssp (opencl)
  The single-source shortest path (https://github.com/chai-benchmarks/chai)

### stddev (cuda)
  Standard deviation (https://github.com/rapidsai/raft)

### stencil1d (cuda)
  1D stencil (https://www.olcf.ornl.gov/wp-content/uploads/2019/12/02-CUDA-Shared-Memory.pdf)

### stencil3d (cuda)
  3D stencil (https://github.com/LLNL/cardioid)

### streamcluster (opencl)
  Streamcluster in the Rodinia benchmark suite (http://lava.cs.virginia.edu/Rodinia/download_links.htm)

### su3 (sycl)
  Lattice QCD SU(3) matrix-matrix multiply microbenchmark (https://gitlab.com/NERSC/nersc-proxies/su3_bench)

### surfel (cuda)
  Surfel rendering (https://github.com/jstraub/cudaPcl)

### svd3x3 (cuda)
  Compute the singular value decomposition of 3x3 matrices (https://github.com/kuiwuchn/3x3_SVD_CUDA)

### sw4ck (cuda)
  SW4 curvilinear kernels are five stencil kernels that account for ~50% of the solution time in SW4 (https://github.com/LLNL/SW4CK)

### testSNAP (openmp)
  A proxy for the SNAP force calculation in the LAMMPS molecular dynamics package (https://github.com/FitSNAP/TestSNAP)

### thomas (cuda)
  Solve tridiagonal systems of equations using the Thomas algorithm (https://pm.bsc.es/gitlab/run-math/cuThomasBatch/tree/master)

### threadfence (cuda)
  Memory fence function (https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#memory-fence-functions)

### tissue (cuda)
  Accumulate contributions of tissue source strengths and previous solute levels to current tissue solute levels (https://github.com/secomb/GreensTD19_GPU)

### tonemapping (opencl)
  Tone mapping (https://github.com/OpenCL/AMD_APP_samples)

### transpose (cuda)
  Tensor transposition (https://github.com/Jokeren/GPA-Benchmark/tree/master/ExaTENSOR)

### triad (opencl)
  Triad in the SHOC benchmark suite(https://github.com/vetter/shoc/)

### tridiagonal (opencl)
  Matrix solvers for large number of small independent tridiagonal linear systems(http://developer.download.nvidia.com/compute/cuda/3_0/sdk/website/OpenCL/website/samples.html)

### tsa (cuda)
  Trotter-Suzuki approximation (https://bitbucket.org/zzzoom/trottersuzuki/src/master/)

### tsp (cuda)
  Solving the symmetric traveling salesman problem with iterative hill climbing (https://userweb.cs.txstate.edu/~burtscher/research/TSP_GPU/) 

### urng (opencl)
  Uniform random noise generator (https://github.com/OpenCL/AMD_APP_samples)
  
### vmc (cuda)
  Computes expectation values (6D integrals) associated with the helium atom (https://github.com/wadejong/Summer-School-Materials/tree/master/Examples/vmc)

### winograd (cuda)
  Winograd convolution (https://github.com/ChenyangZhang-cs/iMLBench)

### wyllie (cuda)
  List ranking with Wyllie's algorithm (Rehman, M. & Kothapalli, Kishore & Narayanan, P.. (2009). Fast and Scalable List Ranking on the GPU. Proceedings of the International Conference on Supercomputing. 235-243. 10.1145/1542275.1542311.)

### xlqc (cuda)
  Hartree-Fock self-consistent-field (SCF) calculation of H2O (https://github.com/recoli/XLQC) 

### xsbench (opencl)
  A proxy application for full neutron transport application like OpenMC (https://github.com/ANL-CESAR/XSBench/)


## Developer
Authored and maintained by Zheming Jin (https://github.com/zjin-lcf) 

## Acknowledgement
Anton Gorshkov, Bernhard Esslinger, Bert de Jong, Chengjian Liu, Chris Knight, David Oro, Douglas Franz, Edson Borin, Gabriell Araujo, Ian Karlin, Istvan Reguly, Jason Lau, Jeff Hammond, Wayne Joubert, Jakub Chlanda, Jiya Su, John Tramm, Ju Zheng, Martin Burtscher, Matthias Noack, Michael Kruse, Michel Migdal, Mike Giles, Mohammed Alser, Muhammad Haseeb, Muaaz Awan, Nevin Liber, Nicholas Miller, Pedro Valero Lara, Piotr Różański, Rahulkumar Gayatri, Shaoyi Peng, Robert Harrison, Rodrigo Vimieiro, Tadej Ciglarič, Thomas Applencourt, Tiago Carneiro, Tobias Baumann, Usman Roshan, Ye Luo, Yongbin Gu, Zhe Chen 

Results presented were obtained using the Chameleon testbed supported by the National Science Foundation, JLSE testbeds at Argonne National Laboratory, and the Intel<sup>®</sup> DevCloud. The project also used resources at the Experimental Computing Laboratory (ExCL) at Oak Ridge National Laboratory, which is supported by the Office of Science of the U.S. Department of Energy under Contract No. DE-AC05-00OR22725.	
