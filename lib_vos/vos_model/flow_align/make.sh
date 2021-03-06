#!/usr/bin/env bash

CUDA_PATH=/usr/local/cuda/

cd src
echo "Compiling my_lib kernels by nvcc..."
nvcc -c -o flow_align_cuda_kernel.cu.o flow_align_cuda_kernel.cu -x cu -Xcompiler -fPIC -arch=sm_61

cd ../
python3 build.py
