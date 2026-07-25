#pragma once

#include <random>

class RandomGenerator
{
public:
    RandomGenerator();

    float NextGaussian();

private:
    std::mt19937 mGenerator;
    std::normal_distribution<float> mDistribution;
};