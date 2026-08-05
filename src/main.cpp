#include "GpuMonteCarlo.h"

#include <cuda_runtime.h>

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

    std::cout
        << "GPU Monte Carlo Scale Test\n";

    std::cout
        << "Total simulations: "
        << totalSimulations
        << '\n';

    std::cout
        << "Batch size: "
        << batchSize
        << "\n\n";

    // Initialize CUDA before timing.
    cudaFree(nullptr);

    const auto start =
        std::chrono::steady_clock::now();

    const auto benchmark =
        RunBatchedGpuMonteCarlo(
            initialPrice,
            expectedReturn,
            volatility,
            tradingDays,
            totalSimulations,
            batchSize
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

    const double simulationsPerSecond =
        static_cast<double>(
            benchmark.totalSimulations
        )
        / (
            endToEndMilliseconds
            / 1000.0
        );

    std::cout
        << "\n"
        << std::fixed
        << std::setprecision(2);

    std::cout
        << "Total end-to-end time: "
        << endToEndMilliseconds
        << " ms\n";

    std::cout
        << "Total kernel time: "
        << benchmark.totalKernelMilliseconds
        << " ms\n";

    std::cout
        << "Throughput: "
        << simulationsPerSecond
        << " simulations/second\n\n";

    std::cout
        << "Mean final price: "
        << benchmark.meanFinalPrice
        << '\n';

    std::cout
        << "Minimum final price: "
        << benchmark.minimumFinalPrice
        << '\n';

    std::cout
        << "Maximum final price: "
        << benchmark.maximumFinalPrice
        << '\n';

    return 0;
}