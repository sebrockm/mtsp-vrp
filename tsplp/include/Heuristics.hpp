#pragma once

#include <xtensor/xarray.hpp>
#include <xtensor/xtensor.hpp>

#include <tuple>
#include <vector>

namespace tsplp
{
class DependencyGraph;

std::tuple<std::vector<std::vector<size_t>>, double> ExploitFractionalSolution(
    xt::xarray<double> fractionalSolution, xt::xarray<double> weights,
    const xt::xtensor<size_t, 1>& startPositions, const xt::xtensor<size_t, 1>& endPositions,
    const DependencyGraph& dependencies, std::chrono::milliseconds timeout);

std::tuple<std::vector<std::vector<size_t>>, double> NearestInsertion(
    xt::xarray<double> weights, const xt::xtensor<size_t, 1>& startPositions,
    const xt::xtensor<size_t, 1>& endPositions, const DependencyGraph& dependencies,
    std::chrono::milliseconds timeout);

std::tuple<std::vector<std::vector<size_t>>, double> TwoOptPaths(
    std::vector<std::vector<size_t>> paths, xt::xarray<double> weights,
    const DependencyGraph& dependencies);
}
