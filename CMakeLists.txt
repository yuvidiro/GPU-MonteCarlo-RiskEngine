cmake_minimum_required(VERSION 3.18)

project(
    GPUMonteCarloRiskEngine
    VERSION 1.0
    LANGUAGES CXX CUDA
)

set(CMAKE_CUDA_ARCHITECTURES 86)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(OpenMP REQUIRED)
find_package(CUDAToolkit REQUIRED)

add_executable(
    MonteCarlo
    src/main.cpp
    src/MonteCarloEngine.cpp
    src/RandomGenerator.cpp
    src/GpuMonteCarlo.cu
)

target_include_directories(
    MonteCarlo
    PRIVATE
    include
)

target_link_libraries(
    MonteCarlo
    PRIVATE
    OpenMP::OpenMP_CXX
    CUDA::curand
)

set_target_properties(
    MonteCarlo
    PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)