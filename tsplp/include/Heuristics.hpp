#pragma once

#include "MtspModel.hpp"

#include <xtensor/xarray.hpp>
#include <xtensor/xtensor.hpp>
#include <xtensor/xview.hpp>

#include <tuple>
#include <vector>

namespace tsplp
{
class DependencyGraph;

[[nodiscard]] std::vector<std::vector<size_t>> ExploitFractionalSolution(
    OptimizationMode optimizationMode, xt::xarray<double> fractionalSolution,
    xt::xarray<double> weights, const xt::xtensor<size_t, 1>& startPositions,
    const xt::xtensor<size_t, 1>& endPositions, const DependencyGraph& dependencies,
    std::chrono::steady_clock::time_point endTime);

[[nodiscard]] std::tuple<std::vector<std::vector<size_t>>, double> NearestInsertion(
    OptimizationMode optimizationMode, xt::xarray<double> weights,
    const xt::xtensor<size_t, 1>& startPositions, const xt::xtensor<size_t, 1>& endPositions,
    const DependencyGraph& dependencies, std::chrono::steady_clock::time_point endTime);

[[nodiscard]] std::tuple<std::vector<std::vector<size_t>>, double> TwoOptPaths(
    OptimizationMode optimizationMode, std::vector<std::vector<size_t>> paths,
    xt::xarray<double> weights, const DependencyGraph& dependencies,
    std::chrono::steady_clock::time_point endTime);

[[nodiscard]] double CalculatePathLength(
    const std::vector<size_t>& path, xt::xarray<double> weights);

[[nodiscard]] double CalculateObjective(
    OptimizationMode optimizationMode, const std::vector<std::vector<size_t>>& paths,
    xt::xarray<double> weights);
}
