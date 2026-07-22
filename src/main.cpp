#include <iostream>

#include "Simulation.h"

int main()
{
    Simulation simulation(
        100.0f,
        0.08f,
        0.20f,
        252);

    auto results = simulation.Run(10);

    std::cout
        << "Simulation completed.\n";

    return 0;
}