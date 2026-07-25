#include <iostream>

#include "MonteCarloEngine.h"

int main()
{
    MonteCarloEngine engine(
        100.0f,   // Initial stock price
        0.08f,    // Expected annual return
        0.20f,    // Annual volatility
        252);     // Trading days

    auto prices = engine.SimulateOnePath();

    for (size_t day = 0; day < prices.size(); ++day)
    {
        std::cout
            << "Day "
            << day
            << " : "
            << prices[day]
            << '\n';
    }

    return 0;
}