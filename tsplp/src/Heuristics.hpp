#pragma once

#include <tuple>
#include <vector>
#include <xtensor/xtensor.hpp>

namespace tsplp
{
    std::tuple<std::vector<std::vector<int>>, int> NearestInsertion(
        xt::xtensor<int, 2> const& weights, xt::xtensor<int, 1> startPositions, xt::xtensor<int, 1> endPositions);
}