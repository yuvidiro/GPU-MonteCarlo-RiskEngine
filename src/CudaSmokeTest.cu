#include "CudaSmokeTest.h"

#include <cuda_runtime.h>

#include <cstdlib>
#include <iostream>
#include <vector>

#define CUDA_CHECK(call)                                             \
do                                                                   \
{                                                                    \
    cudaError_t error = (call);                                      \
                                                                     \
    if (error != cudaSuccess)                                        \
    {                                                                \
        std::cerr                                                    \
            << "CUDA error: "                                        \
            << cudaGetErrorString(error)                             \
            << "\nFile: " << __FILE__                               \
            << "\nLine: " << __LINE__                               \
            << '\n';                                                 \
                                                                     \
        std::exit(EXIT_FAILURE);                                     \
    }                                                                \
} while (0)

__global__
void SquareKernel(
    const int* input,
    int* output,
    int count)
{
    const int index =
        blockIdx.x * blockDim.x + threadIdx.x;

    if (index < count)
    {
        output[index] =
            input[index] * input[index];
    }
}

void RunCudaSmokeTest()
{
    constexpr int count = 10;

    std::vector<int> hostInput(count);
    std::vector<int> hostOutput(count, -1);

    for (int i = 0; i < count; ++i)
    {
        hostInput[i] = i;
    }

    int* deviceInput = nullptr;
    int* deviceOutput = nullptr;

    const std::size_t bytes =
        count * sizeof(int);

    CUDA_CHECK(
        cudaMalloc(
            reinterpret_cast<void**>(&deviceInput),
            bytes
        )
    );

    CUDA_CHECK(
        cudaMalloc(
            reinterpret_cast<void**>(&deviceOutput),
            bytes
        )
    );

    CUDA_CHECK(
        cudaMemcpy(
            deviceInput,
            hostInput.data(),
            bytes,
            cudaMemcpyHostToDevice
        )
    );

    constexpr int threadsPerBlock = 256;

    const int blocks =
        (count + threadsPerBlock - 1)
        / threadsPerBlock;

    SquareKernel<<<
        blocks,
        threadsPerBlock
    >>>(
        deviceInput,
        deviceOutput,
        count
    );

    // Check whether the kernel launch was accepted.
    CUDA_CHECK(
        cudaGetLastError()
    );

    // Wait for the GPU and catch runtime errors.
    CUDA_CHECK(
        cudaDeviceSynchronize()
    );

    CUDA_CHECK(
        cudaMemcpy(
            hostOutput.data(),
            deviceOutput,
            bytes,
            cudaMemcpyDeviceToHost
        )
    );

    for (int i = 0; i < count; ++i)
    {
        std::cout
            << hostInput[i]
            << " squared = "
            << hostOutput[i]
            << '\n';
    }

    CUDA_CHECK(
        cudaFree(deviceInput)
    );

    CUDA_CHECK(
        cudaFree(deviceOutput)
    );
}