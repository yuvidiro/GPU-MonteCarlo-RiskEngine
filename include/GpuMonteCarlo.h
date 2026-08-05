#pragma once

#include <cstddef>
#include <vector>

struct GpuBenchmarkResult
{
    std::vector<float> results;

    float kernelMilliseconds;
};

GpuBenchmarkResult RunGpuMonteCarlo(
    float initialPrice,
    float expectedReturn,
    float volatility,
    int tradingDays,
    std::size_t numSimulations
);