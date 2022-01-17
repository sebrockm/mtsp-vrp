#pragma once

#include <tuple>
#include <vector>
#include <xtensor/xtensor.hpp>

namespace tsplp
{
    std::tuple<std::vector<std::vector<size_t>>, int> NearestInsertion(
        const xt::xtensor<int, 2>& weights, const xt::xtensor<size_t, 1>& startPositions, const xt::xtensor<size_t, 1>& endPositions);
}