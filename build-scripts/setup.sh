#!/bin/bash

BUILD_DIR=build

if [ -d "$BUILD_DIR" ]; then 
  rm -Rf $BUILD_DIR
fi

mkdir $BUILD_DIR &&
cd $BUILD_DIR &&
../build-scripts/build.sh RelWithDebInfo ../sources/llvm on &&
cmake --build . --target install &&
cd ..
 
export PATH=$(pwd)/build/install/bin:$PATH 
