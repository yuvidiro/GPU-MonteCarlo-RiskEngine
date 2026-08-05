#include "GpuMonteCarlo.h"

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <utility>
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
void MonteCarloKernel(
    float* results,
    float initialPrice,
    float expectedReturn,
    float volatility,
    int tradingDays,
    std::size_t numSimulations,
    unsigned long long seed,
    std::size_t simulationOffset)
{
    const std::size_t localSimulationId =
        static_cast<std::size_t>(blockIdx.x)
        * blockDim.x
        + threadIdx.x;

    if (localSimulationId >= numSimulations)
    {
        return;
    }

    // Each batch must use different random sequences.
    const std::size_t globalSimulationId =
        simulationOffset
        + localSimulationId;

    curandStatePhilox4_32_10_t randomState;

    curand_init(
        seed,
        globalSimulationId,
        0,
        &randomState
    );

    constexpr float dt =
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

    results[localSimulationId] =
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
        12345ULL,
        0
    );

    CUDA_CHECK(
        cudaGetLastError()
    );

    CUDA_CHECK(
        cudaEventRecord(
            kernelStop
        )
    );

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
    // Allocate the largest required GPU buffer once.
    float* deviceResults =
        nullptr;

    const std::size_t deviceBytes =
        batchSize
        * sizeof(float);

    CUDA_CHECK(
        cudaMalloc(
            reinterpret_cast<void**>(
                &deviceResults
            ),
            deviceBytes
        )
    );

    // Reuse one CPU buffer for every batch.
    std::vector<float> hostResults(
        batchSize
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

    double totalSum =
        0.0;

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

        const int blocks =
            static_cast<int>(
                (
                    currentBatchSize
                    + threadsPerBlock
                    - 1
                )
                / threadsPerBlock
            );

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
            currentBatchSize,
            12345ULL,
            completedSimulations
        );

        CUDA_CHECK(
            cudaGetLastError()
        );

        CUDA_CHECK(
            cudaEventRecord(
                kernelStop
            )
        );

        CUDA_CHECK(
            cudaEventSynchronize(
                kernelStop
            )
        );

        float batchKernelMilliseconds =
            0.0f;

        CUDA_CHECK(
            cudaEventElapsedTime(
                &batchKernelMilliseconds,
                kernelStart,
                kernelStop
            )
        );

        totalKernelMilliseconds +=
            batchKernelMilliseconds;

        const std::size_t currentBatchBytes =
            currentBatchSize
            * sizeof(float);

        CUDA_CHECK(
            cudaMemcpy(
                hostResults.data(),
                deviceResults,
                currentBatchBytes,
                cudaMemcpyDeviceToHost
            )
        );

        for (
            std::size_t i = 0;
            i < currentBatchSize;
            ++i)
        {
            const float finalPrice =
                hostResults[i];

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

    // Free GPU memory once after every batch finishes.
    CUDA_CHECK(
        cudaFree(
            deviceResults
        )
    );

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