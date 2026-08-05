#pragma once

#include <cstddef>
#include <vector>

struct GpuBenchmarkResult
{
    std::vector<float> results;

    float kernelMilliseconds;
};

struct GpuBatchBenchmarkResult
{
    std::size_t totalSimulations;

    double meanFinalPrice;

    float minimumFinalPrice;

    float maximumFinalPrice;

    double totalKernelMilliseconds;
};

GpuBenchmarkResult RunGpuMonteCarlo(
    float initialPrice,
    float expectedReturn,
    float volatility,
    int tradingDays,
    std::size_t numSimulations,
    int threadsPerBlock = 256
);

GpuBatchBenchmarkResult RunBatchedGpuMonteCarlo(
    float initialPrice,
    float expectedReturn,
    float volatility,
    int tradingDays,
    std::size_t totalSimulations,
    std::size_t batchSize,
    int threadsPerBlock = 256
);