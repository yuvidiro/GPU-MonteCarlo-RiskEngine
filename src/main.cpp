#include "GpuMonteCarlo.h"
#include "MonteCarloEngine.h"

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

    constexpr std::size_t numSimulations =
        1'000'000;

    std::cout
        << "Monte Carlo CPU vs GPU Benchmark\n";

    std::cout
        << "Simulations: "
        << numSimulations
        << "\n\n";

    // -------------------------
    // CPU benchmark
    // -------------------------

    MonteCarloEngine cpuEngine(
        initialPrice,
        expectedReturn,
        volatility,
        tradingDays
    );

    const auto cpuStart =
        std::chrono::steady_clock::now();

    auto cpuResults =
        cpuEngine.Run(
            numSimulations
        );

    const auto cpuEnd =
        std::chrono::steady_clock::now();

    const double cpuMilliseconds =
        std::chrono::duration<
            double,
            std::milli
        >(
            cpuEnd - cpuStart
        ).count();

    // -------------------------
    // CUDA warm-up
    // -------------------------

    cudaFree(nullptr);

    // -------------------------
    // GPU end-to-end benchmark
    // -------------------------

    const auto gpuStart =
        std::chrono::steady_clock::now();

    auto gpuBenchmark =
        RunGpuMonteCarlo(
            initialPrice,
            expectedReturn,
            volatility,
            tradingDays,
            numSimulations
        );

    const auto gpuEnd =
        std::chrono::steady_clock::now();

    const double gpuMilliseconds =
        std::chrono::duration<
            double,
            std::milli
        >(
            gpuEnd - gpuStart
        ).count();

    // -------------------------
    // Speedup calculations
    // -------------------------

    const double endToEndSpeedup =
        cpuMilliseconds
        / gpuMilliseconds;

    const double kernelOnlySpeedup =
        cpuMilliseconds
        / gpuBenchmark.kernelMilliseconds;

    // -------------------------
    // Print results
    // -------------------------

    std::cout
        << std::fixed
        << std::setprecision(2);

    std::cout
        << "CPU time: "
        << cpuMilliseconds
        << " ms\n";

    std::cout
        << "GPU end-to-end time: "
        << gpuMilliseconds
        << " ms\n";

    std::cout
        << "GPU kernel time: "
        << gpuBenchmark.kernelMilliseconds
        << " ms\n\n";

    std::cout
        << "End-to-end GPU speedup: "
        << endToEndSpeedup
        << "x\n";

    std::cout
        << "Kernel-only GPU speedup: "
        << kernelOnlySpeedup
        << "x\n\n";

    std::cout
        << "CPU results: "
        << cpuResults.size()
        << '\n';

    std::cout
        << "GPU results: "
        << gpuBenchmark.results.size()
        << '\n';

    return 0;
}