#include "GpuMonteCarlo.h"

#include <cuda_runtime.h>

#include <array>
#include <chrono>
#include <iomanip>
#include <iostream>

int main()
{
    constexpr float initialPrice =
        100.0f;

    constexpr float expectedReturn =
        0.08f;

    constexpr float volatility =
        0.20f;

    constexpr int tradingDays =
        252;

    constexpr std::size_t totalSimulations =
        100'000'000;

    constexpr std::size_t batchSize =
        10'000'000;

    constexpr std::array<int, 3>
        blockSizes =
        {
            128,
            256,
            512
        };

    std::cout
        << "CUDA Block Size Benchmark\n";

    std::cout
        << "Total simulations: "
        << totalSimulations
        << '\n';

    std::cout
        << "Batch size: "
        << batchSize
        << "\n\n";

    // Initialize CUDA before benchmarking.
    cudaFree(nullptr);

    std::cout
        << std::fixed
        << std::setprecision(2);

    std::cout
        << "Threads/block"
        << "\tKernel ms"
        << "\tEnd-to-end ms"
        << "\tThroughput (M/s)"
        << "\n";

    std::cout
        << "------------------------------------------------"
        << "\n";

    for (
        const int threadsPerBlock
        : blockSizes)
    {
        std::cout
            << "\nTesting "
            << threadsPerBlock
            << " threads per block\n";

        const auto start =
            std::chrono::steady_clock::now();

        const auto benchmark =
            RunBatchedGpuMonteCarlo(
                initialPrice,
                expectedReturn,
                volatility,
                tradingDays,
                totalSimulations,
                batchSize,
                threadsPerBlock
            );

        const auto end =
            std::chrono::steady_clock::now();

        const double endToEndMilliseconds =
            std::chrono::duration<
                double,
                std::milli
            >(
                end - start
            ).count();

        const double throughput =
            static_cast<double>(
                benchmark.totalSimulations
            )
            / (
                endToEndMilliseconds
                / 1000.0
            );

        const double throughputMillions =
            throughput
            / 1'000'000.0;

        std::cout
            << "\n"
            << threadsPerBlock
            << "\t\t"
            << benchmark.totalKernelMilliseconds
            << "\t\t"
            << endToEndMilliseconds
            << "\t\t"
            << throughputMillions
            << '\n';

        std::cout
            << "Mean: "
            << benchmark.meanFinalPrice
            << ", Min: "
            << benchmark.minimumFinalPrice
            << ", Max: "
            << benchmark.maximumFinalPrice
            << "\n";
    }

    return 0;
}