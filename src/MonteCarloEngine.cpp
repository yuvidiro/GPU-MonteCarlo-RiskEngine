#include "MonteCarloEngine.h"

#include <cmath>

MonteCarloEngine::MonteCarloEngine(
    float initialPrice,
    float expectedReturn,
    float volatility,
    int tradingDays)
    :
    mInitialPrice(initialPrice),
    mExpectedReturn(expectedReturn),
    mVolatility(volatility),
    mTradingDays(tradingDays)
{
}

std::vector<float> MonteCarloEngine::SimulateOnePath()
{
    std::vector<float> prices;

    // Reserve memory to avoid reallocations
    prices.reserve(mTradingDays + 1);

    float price = mInitialPrice;

    // Store initial price (Day 0)
    prices.push_back(price);

    // Time step (1 trading day)
    const float dt = 1.0f / 252.0f;

    // Constant drift term
    const float drift =
        (mExpectedReturn - 0.5f * mVolatility * mVolatility) * dt;

    // Constant diffusion term
    const float diffusion =
        mVolatility * std::sqrt(dt);

    for (int day = 0; day < mTradingDays; ++day)
    {
        // Standard normal random number
        float z = mRandom.NextGaussian();

        // Geometric Brownian Motion update
        price *= std::exp(drift + diffusion * z);

        prices.push_back(price);
    }

    return prices;
}