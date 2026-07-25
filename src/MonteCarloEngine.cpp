#include "MonteCarloEngine.h"

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

    float price = mInitialPrice;

    prices.push_back(price);

    for(int day = 0; day < mTradingDays; day++)
    {
        float z = mRandom.NextGaussian();

        // GBM update comes here

        prices.push_back(price);
    }

    return prices;
}