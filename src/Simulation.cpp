#include "Simulation.h"

Simulation::Simulation(
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

std::vector<float>
Simulation::Run(std::size_t numSimulations)
{
    return {};
}