#pragma once

#include <vector>

#include "RandomGenerator.h"

class MonteCarloEngine
{
public:

    MonteCarloEngine(
        float initialPrice,
        float expectedReturn,
        float volatility,
        int tradingDays);

    std::vector<float> SimulateOnePath();

private:

    float mInitialPrice;
    float mExpectedReturn;
    float mVolatility;
    int mTradingDays;

    RandomGenerator mRandom;
};