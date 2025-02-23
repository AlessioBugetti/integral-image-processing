cmake_minimum_required(VERSION 3.13)
project(integral_image LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)

find_package(CUDAToolkit REQUIRED)
set(CUDA_ARCHITECTURES 86)

add_executable(integral_image main.cu)