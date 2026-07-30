#pragma once

#include <cstddef>
#include <vector>

class MonteCarloEngine
{
public:
    MonteCarloEngine(
        float initialPrice,
        float expectedReturn,
        float volatility,
        int tradingDays);

    std::vector<float> Run(
        std::size_t numSimulations);

private:
    float mInitialPrice;
    float mExpectedReturn;
    float mVolatility;
    int mTradingDays;
};