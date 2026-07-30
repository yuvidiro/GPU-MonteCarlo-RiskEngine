#include "MonteCarloEngine.h"
#include "RandomGenerator.h"
#include <cmath>
#include <cstdint>
#include <omp.h>

MonteCarloEngine::MonteCarloEngine(
    float initialPrice,
    float expectedReturn,
    float volatility,
    int tradingDays)
    : mInitialPrice(initialPrice),
      mExpectedReturn(expectedReturn),
      mVolatility(volatility),
      mTradingDays(tradingDays)
{
}

std::vector<float> MonteCarloEngine::Run(
    std::size_t numSimulations)
{
    std::vector<float> results(numSimulations);

    const float dt = 1.0f / 252.0f;

    const float drift =
        (mExpectedReturn
        - 0.5f * mVolatility * mVolatility) * dt;

    const float diffusion =
        mVolatility * std::sqrt(dt);

#pragma omp parallel
    {
        const int threadId = omp_get_thread_num();

        RandomGenerator random(
            12345u + static_cast<std::uint32_t>(threadId)
        );

#pragma omp for
        for (long long sim = 0;
             sim < static_cast<long long>(numSimulations);
             ++sim)
        {
            float price = mInitialPrice;

            for (int day = 0;
                 day < mTradingDays;
                 ++day)
            {
                const float z =
                    random.NextGaussian();

                price *= std::exp(
                    drift + diffusion * z
                );
            }

            results[
                static_cast<std::size_t>(sim)
            ] = price;
        }
    }

    return results;
}