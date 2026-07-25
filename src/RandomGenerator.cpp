#include "RandomGenerator.h"

RandomGenerator::RandomGenerator()
    : mGenerator(std::random_device{}()),
      mDistribution(0.0f, 1.0f)
{
}

float RandomGenerator::NextGaussian()
{
    return mDistribution(mGenerator);
}