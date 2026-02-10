cmake_minimum_required(VERSION 3.22.1)
project(RFANN LANGUAGES CXX CUDA)

include(GNUInstallDirs)
include(CheckCXXCompilerFlag)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

SET( CMAKE_CXX_FLAGS  "-O3 -march=native -lrt -std=c++11 -DHAVE_CXX0X -fpic -w -fopenmp -ftree-vectorize -ftree-vectorizer-verbose=0" )

include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

add_subdirectory(tests)