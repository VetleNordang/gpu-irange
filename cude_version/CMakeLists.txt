cmake_minimum_required(VERSION 3.18)
project(HelloCUDA LANGUAGES CXX CUDA)

add_executable(hello hello.cu)
set_target_properties(hello PROPERTIES CUDA_ARCHITECTURES "75;80;86")
