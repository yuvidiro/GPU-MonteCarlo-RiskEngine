#include "RandomGenerator.h"

RandomGenerator::RandomGenerator(std::uint32_t seed)
    : mGenerator(seed),
      mDistribution(0.0f, 1.0f)
{
}

float RandomGenerator::NextGaussian()
{
    return mDistribution(mGenerator);
}