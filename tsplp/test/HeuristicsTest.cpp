#include "Heuristics.hpp"

#include <catch2/catch.hpp>

#include <vector>

TEST_CASE("path length", "[Heuristics]")
{
    // clang-format off
    xt::xarray<double> weights =
    {
        { 1, 2, 3, 4 },
        { 2, 4, 6, 8 },
        { 4, 5, 6, 7 },
        { 0, 1, 2, 3 }
    };
    // clang-format on

    REQUIRE(tsplp::CalculatePathLength({ 0, 1, 2, 3, 0 }, weights) == 15);
    REQUIRE(tsplp::CalculatePathLength({ 2, 0, 3, 1, 2 }, weights) == 15);
    REQUIRE(tsplp::CalculatePathLength({ 1, 2, 3, 0 }, weights) == 13);
    REQUIRE(tsplp::CalculatePathLength({ 3, 2, 1 }, weights) == 7);
    REQUIRE(tsplp::CalculatePathLength({ 0, 3 }, weights) == 4);
    REQUIRE(tsplp::CalculatePathLength({ 3 }, weights) == 0);
    REQUIRE(tsplp::CalculatePathLength({}, weights) == 0);
}

TEST_CASE("objective A==1", "[Heuristics]")
{
    // clang-format off
    xt::xarray<double> weights =
    {
        { 1, 2, 3, 4 },
        { 2, 4, 6, 8 },
        { 4, 5, 6, 7 },
        { 0, 1, 2, 3 }
    };
    // clang-format on

    std::vector<std::vector<size_t>> paths1 = { { 0, 1, 2, 3 } };
    REQUIRE(tsplp::CalculateObjective(tsplp::OptimizationMode::Sum, paths1, weights) == 15);
    REQUIRE(tsplp::CalculateObjective(tsplp::OptimizationMode::Max, paths1, weights) == 15);
}

TEST_CASE("objective A==2 empty", "[Heuristics]")
{
    // clang-format off
    xt::xarray<double> weights =
    {
        { 1, 2, 3, 4 },
        { 2, 4, 6, 8 },
        { 4, 5, 6, 7 },
        { 0, 1, 2, 3 }
    };
    // clang-format on

    std::vector<std::vector<size_t>> paths1 = { {} };
    REQUIRE(tsplp::CalculateObjective(tsplp::OptimizationMode::Sum, paths1, weights) == 0);
    REQUIRE(tsplp::CalculateObjective(tsplp::OptimizationMode::Max, paths1, weights) == 0);
}

TEST_CASE("objective A==2", "[Heuristics]")
{
    // clang-format off
    xt::xarray<double> weights =
    {
        { 1, 2, 3, 4 },
        { 2, 4, 6, 8 },
        { 4, 5, 6, 7 },
        { 0, 1, 2, 3 }
    };
    // clang-format on

    std::vector<std::vector<size_t>> paths1 = { { 0, 1, 2, 3 }, { 0, 2, 1, 3 } };
    REQUIRE(tsplp::CalculateObjective(tsplp::OptimizationMode::Sum, paths1, weights) == 15 + 16);
    REQUIRE(tsplp::CalculateObjective(tsplp::OptimizationMode::Max, paths1, weights) == 16);
}

TEST_CASE("objective A==4", "[Heuristics]")
{
    // clang-format off
    xt::xarray<double> weights =
    {
        { 1, 2, 3, 4 },
        { 2, 4, 6, 8 },
        { 4, 5, 6, 7 },
        { 0, 1, 2, 3 }
    };
    // clang-format on

    std::vector<std::vector<size_t>> paths1 = { { 0, 1, 2, 3 }, { 0, 2, 1, 3 }, { 2, 0 }, { 1 } };
    REQUIRE(
        tsplp::CalculateObjective(tsplp::OptimizationMode::Sum, paths1, weights)
        == 15 + 16 + 4 + 0);
    REQUIRE(tsplp::CalculateObjective(tsplp::OptimizationMode::Max, paths1, weights) == 16);
}