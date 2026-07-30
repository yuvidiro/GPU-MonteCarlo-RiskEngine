#pragma once

#include <cstdint>
#include <random>

class RandomGenerator
{
public:
    explicit RandomGenerator(std::uint32_t seed);

    float NextGaussian();

private:
    std::mt19937 mGenerator;
    std::normal_distribution<float> mDistribution;
};