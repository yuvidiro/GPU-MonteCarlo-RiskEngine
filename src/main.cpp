#include <iostream>

#include "RandomGenerator.h"
#include "MonteCarloEngine.h"

int main()
{
   MonteCarloEngine engine(
    100.0f,
    0.08f,
    0.20f,
    252);

auto prices = engine.SimulateOnePath();

for(float price : prices)
{
    std::cout << price << '\n';
}
}