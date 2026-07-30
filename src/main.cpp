#include "GpuMonteCarlo.h"

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
        10000;

    auto results =
        RunGpuMonteCarlo(
            initialPrice,
            expectedReturn,
            volatility,
            tradingDays,
            numSimulations
        );

    for (
        std::size_t i = 0;
        i < 10;
        ++i)
    {
        std::cout
            << "GPU Simulation "
            << i
            << " : "
            << results[i]
            << '\n';
    }

    std::cout
        << "\nTotal GPU Simulations = "
        << results.size()
        << '\n';

    return 0;
}