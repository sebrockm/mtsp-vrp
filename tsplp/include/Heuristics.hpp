#pragma once

#include <tuple>
#include <vector>
#include <xtensor/xtensor.hpp>

namespace tsplp
{
    std::tuple<std::vector<std::vector<int>>, int> NearestInsertion(
        const xt::xtensor<int, 2>& weights, const xt::xtensor<int, 1>& startPositions, const xt::xtensor<int, 1>& endPositions);
}