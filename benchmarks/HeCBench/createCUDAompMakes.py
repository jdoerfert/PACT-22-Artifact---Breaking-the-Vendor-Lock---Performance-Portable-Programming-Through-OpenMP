import glob
import sys
import shutil
from pathlib import Path
import re

def tranformMakeToCuda(origMake):
  cudaMake = []
  cudaOMPMake = []
  compiler  = re.compile('(CC\s.+=\s+nvcc)')
  cflags = re.compile('(CFLAGS\s+:=\s+)')
  lflags = re.compile('(LDFLAGS\s+=\s+)')
  found_cc = False
  found_lf = False
  found_cf = False

  for l in origMake:
    cc = compiler.match(l)
    cf = cflags.match(l)
    lf = lflags.match(l)
    if cc is not None:
      cudaMake.append('CC = clang++\n')
      cudaOMPMake.append('CC = clang++\n')
      found_cc = True
    elif cf is not None:
      cudaMake.append('CFLAGS := -I${CUDA_HOME}/include -std=c++14 -Wall --cuda-gpu-arch=sm_70\n')
      cudaOMPMake.append('CFLAGS :=-std=c++14 --cuda-gpu-arch=sm_70 -cudaomp\n')
      found_cf = True
    elif lf is not None:
      cudaMake.append('LDFLAGS =-L${CUDA_HOME}/lib64 -lcudart -lcuda \n')
      cudaOMPMake.append('LDFLAGS= -ldl -lrt -pthread -lomptarget -lomp\n')
      found_lf = True
    else:
      cudaMake.append(l)
      cudaOMPMake.append(l)
  return (cudaMake, cudaOMPMake, found_lf & found_cf & found_cc)

def main(args):
  processed = []
  for g in glob.glob('./*-cuda/'):
    origMakefile = Path(f'{g}Makefile')
    if origMakefile.is_file():
      with open(f'{g}/Makefile', 'r') as fd:
        origMake = fd.readlines()
      cudaMake, ompMake, correct = tranformMakeToCuda(origMake)
      if not correct:
        print(f'I could not process {g}')
      else:
        processed.append(g)
        with open(f'{g}/Makefile.cuda', 'w') as fd:
          fd.writelines(cudaMake)
        with open(f'{g}/Makefile.cudaomp', 'w') as fd:
          fd.writelines(ompMake)
    else:
      print('File does not exist', g)

  with open('processed.txt', 'w') as fd:
    fd.writelines([f'{v}\n' for v in processed])


main(sys.argv)
