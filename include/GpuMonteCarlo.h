#pragma once

#include <cstddef>
#include <vector>

std::vector<float> RunGpuMonteCarlo(
    float initialPrice,
    float expectedReturn,
    float volatility,
    int tradingDays,
    std::size_t numSimulations
);