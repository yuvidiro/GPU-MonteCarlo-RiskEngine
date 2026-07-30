#include "GpuMonteCarlo.h"

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <cmath>
#include <cstdlib>
#include <iostream>

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
void MonteCarloKernel(
    float* results,
    float initialPrice,
    float expectedReturn,
    float volatility,
    int tradingDays,
    std::size_t numSimulations,
    unsigned long long seed)
{
    const std::size_t simulationId =
        static_cast<std::size_t>(
            blockIdx.x
        ) * blockDim.x
        + threadIdx.x;

    if (simulationId >= numSimulations)
    {
        return;
    }

    curandStatePhilox4_32_10_t randomState;

    curand_init(
        seed,
        simulationId,
        0,
        &randomState
    );

    const float dt =
        1.0f / 252.0f;

    const float drift =
        (
            expectedReturn
            - 0.5f
            * volatility
            * volatility
        ) * dt;

    const float diffusion =
        volatility
        * sqrtf(dt);

    float price =
        initialPrice;

    for (
        int day = 0;
        day < tradingDays;
        ++day
    )
    {
        const float z =
            curand_normal(
                &randomState
            );

        price *= expf(
            drift
            + diffusion * z
        );
    }

    results[simulationId] =
        price;
}

std::vector<float> RunGpuMonteCarlo(
    float initialPrice,
    float expectedReturn,
    float volatility,
    int tradingDays,
    std::size_t numSimulations)
{
    std::vector<float> hostResults(
        numSimulations
    );

    float* deviceResults =
        nullptr;

    const std::size_t bytes =
        numSimulations
        * sizeof(float);

    CUDA_CHECK(
        cudaMalloc(
            reinterpret_cast<void**>(
                &deviceResults
            ),
            bytes
        )
    );

    constexpr int threadsPerBlock =
        256;

    const int blocks =
        static_cast<int>(
            (
                numSimulations
                + threadsPerBlock
                - 1
            )
            / threadsPerBlock
        );

    MonteCarloKernel<<<
        blocks,
        threadsPerBlock
    >>>(
        deviceResults,
        initialPrice,
        expectedReturn,
        volatility,
        tradingDays,
        numSimulations,
        12345ULL
    );

    CUDA_CHECK(
        cudaGetLastError()
    );

    CUDA_CHECK(
        cudaDeviceSynchronize()
    );

    CUDA_CHECK(
        cudaMemcpy(
            hostResults.data(),
            deviceResults,
            bytes,
            cudaMemcpyDeviceToHost
        )
    );

    CUDA_CHECK(
        cudaFree(
            deviceResults
        )
    );

    return hostResults;
}