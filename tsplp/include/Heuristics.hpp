#pragma once

#include <xtensor/xarray.hpp>
#include <xtensor/xtensor.hpp>

#include <tuple>
#include <vector>

namespace tsplp
{
class DependencyGraph;

[[nodiscard]] std::tuple<std::vector<std::vector<size_t>>, double> ExploitFractionalSolution(
    xt::xarray<double> fractionalSolution, xt::xarray<double> weights,
    const xt::xtensor<size_t, 1>& startPositions, const xt::xtensor<size_t, 1>& endPositions,
    const DependencyGraph& dependencies, std::chrono::steady_clock::time_point endTime);

[[nodiscard]] std::tuple<std::vector<std::vector<size_t>>, double> NearestInsertion(
    xt::xarray<double> weights, const xt::xtensor<size_t, 1>& startPositions,
    const xt::xtensor<size_t, 1>& endPositions, const DependencyGraph& dependencies,
    std::chrono::steady_clock::time_point endTime);

[[nodiscard]] std::tuple<std::vector<std::vector<size_t>>, double> TwoOptPaths(
    std::vector<std::vector<size_t>> paths, xt::xarray<double> weights,
    const DependencyGraph& dependencies, std::chrono::steady_clock::time_point endTime);
}
