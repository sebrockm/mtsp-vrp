#pragma once

#include <xtensor/xtensor.hpp>

namespace tsplp
{
    xt::xtensor<int, 2> CreateTransitiveDependencies(xt::xtensor<int, 2> weights);
}
