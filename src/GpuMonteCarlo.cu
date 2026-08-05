#include "GpuMonteCarlo.h"

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <utility>
#include <vector>
#include <algorithm>
#include <limits>

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
        static_cast<std::size_t>(blockIdx.x)
        * blockDim.x
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
        ++day)
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

GpuBenchmarkResult RunGpuMonteCarlo(
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

    cudaEvent_t kernelStart;
    cudaEvent_t kernelStop;

    CUDA_CHECK(
        cudaEventCreate(
            &kernelStart
        )
    );

    CUDA_CHECK(
        cudaEventCreate(
            &kernelStop
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

    // Start GPU kernel timing.
    CUDA_CHECK(
        cudaEventRecord(
            kernelStart
        )
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

    // Stop GPU kernel timing.
    CUDA_CHECK(
        cudaEventRecord(
            kernelStop
        )
    );

    // Wait until the GPU reaches the stop event.
    CUDA_CHECK(
        cudaEventSynchronize(
            kernelStop
        )
    );

    float kernelMilliseconds =
        0.0f;

    CUDA_CHECK(
        cudaEventElapsedTime(
            &kernelMilliseconds,
            kernelStart,
            kernelStop
        )
    );

    // Copy results from GPU to CPU.
    CUDA_CHECK(
        cudaMemcpy(
            hostResults.data(),
            deviceResults,
            bytes,
            cudaMemcpyDeviceToHost
        )
    );

    CUDA_CHECK(
        cudaEventDestroy(
            kernelStart
        )
    );

    CUDA_CHECK(
        cudaEventDestroy(
            kernelStop
        )
    );

    CUDA_CHECK(
        cudaFree(
            deviceResults
        )
    );

    return {
        std::move(hostResults),
        kernelMilliseconds
    };
}
GpuBatchBenchmarkResult RunBatchedGpuMonteCarlo(
    float initialPrice,
    float expectedReturn,
    float volatility,
    int tradingDays,
    std::size_t totalSimulations,
    std::size_t batchSize)
{
    double totalSum = 0.0;

    float minimumFinalPrice =
        std::numeric_limits<float>::max();

    float maximumFinalPrice =
        std::numeric_limits<float>::lowest();

    double totalKernelMilliseconds =
        0.0;

    std::size_t completedSimulations =
        0;

    while (
        completedSimulations
        < totalSimulations)
    {
        const std::size_t remaining =
            totalSimulations
            - completedSimulations;

        const std::size_t currentBatchSize =
            std::min(
                batchSize,
                remaining
            );

        auto batchResult =
            RunGpuMonteCarlo(
                initialPrice,
                expectedReturn,
                volatility,
                tradingDays,
                currentBatchSize
            );

        totalKernelMilliseconds +=
            batchResult.kernelMilliseconds;

        for (
            float finalPrice
            : batchResult.results)
        {
            totalSum +=
                finalPrice;

            minimumFinalPrice =
                std::min(
                    minimumFinalPrice,
                    finalPrice
                );

            maximumFinalPrice =
                std::max(
                    maximumFinalPrice,
                    finalPrice
                );
        }

        completedSimulations +=
            currentBatchSize;

        std::cout
            << "Completed "
            << completedSimulations
            << " / "
            << totalSimulations
            << " simulations\n";
    }

    const double meanFinalPrice =
        totalSum
        / static_cast<double>(
            totalSimulations
        );

    return {
        totalSimulations,
        meanFinalPrice,
        minimumFinalPrice,
        maximumFinalPrice,
        totalKernelMilliseconds
    };
}