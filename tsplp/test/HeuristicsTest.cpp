#include "Heuristics.hpp"

#include <catch2/catch.hpp>

#include <chrono>
#include <vector>

using namespace std::chrono_literals;

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

    CHECK(tsplp::CalculatePathLength({ 0, 1, 2, 3, 0 }, weights) == 15);
    CHECK(tsplp::CalculatePathLength({ 2, 0, 3, 1, 2 }, weights) == 15);
    CHECK(tsplp::CalculatePathLength({ 1, 2, 3, 0 }, weights) == 13);
    CHECK(tsplp::CalculatePathLength({ 3, 2, 1 }, weights) == 7);
    CHECK(tsplp::CalculatePathLength({ 0, 3 }, weights) == 4);
    CHECK(tsplp::CalculatePathLength({ 3 }, weights) == 0);
    CHECK(tsplp::CalculatePathLength({}, weights) == 0);
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

    std::vector<std::vector<size_t>> paths = { { 0, 1, 2, 3 } };
    CHECK(tsplp::CalculateObjective(tsplp::OptimizationMode::Sum, paths, weights) == 15);
    CHECK(tsplp::CalculateObjective(tsplp::OptimizationMode::Max, paths, weights) == 15);
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

    std::vector<std::vector<size_t>> paths = { {} };
    CHECK(tsplp::CalculateObjective(tsplp::OptimizationMode::Sum, paths, weights) == 0);
    CHECK(tsplp::CalculateObjective(tsplp::OptimizationMode::Max, paths, weights) == 0);
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

    std::vector<std::vector<size_t>> paths = { { 0, 1, 2, 3 }, { 0, 2, 1, 3 } };
    CHECK(tsplp::CalculateObjective(tsplp::OptimizationMode::Sum, paths, weights) == 15 + 16);
    CHECK(tsplp::CalculateObjective(tsplp::OptimizationMode::Max, paths, weights) == 16);
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

    std::vector<std::vector<size_t>> paths = { { 0, 1, 2, 3 }, { 0, 2, 1, 3 }, { 2, 0 }, { 1 } };
    CHECK(
        tsplp::CalculateObjective(tsplp::OptimizationMode::Sum, paths, weights) == 15 + 16 + 4 + 0);
    CHECK(tsplp::CalculateObjective(tsplp::OptimizationMode::Max, paths, weights) == 16);
}

TEST_CASE("twoopt A==1", "[Heuristics]")
{
    // clang-format off
    xt::xarray<double> weights =
    {
        { 0, 2, 3, 4 },
        { 2, 0, 6, 8 },
        { 4, 5, 0, 7 },
        { 0, 1, 2, 0 }
    };
    // clang-format on

    const std::vector<std::vector<size_t>> paths = { { 0, 1, 3, 2, 0 } };

    {
        const auto [optPaths, improvement] = tsplp::TwoOptPaths(
            tsplp::OptimizationMode::Sum, paths, weights, tsplp::DependencyGraph { weights },
            std::chrono::steady_clock::now() + 1h);
        // expected improvements
        // { { 0, 1, 3, 2, 0 } } => 2 + 8 + 2 + 4
        // { { 0, 3, 1, 2, 0 } } => 4 + 1 + 6 + 4
        // { { 0, 3, 2, 1, 0 } } => 4 + 2 + 5 + 2
        CHECK(improvement == 3);
        CHECK(tsplp::CalculateObjective(tsplp::OptimizationMode::Sum, optPaths, weights) == 13);
    }
    {
        const auto [optPaths, improvement] = tsplp::TwoOptPaths(
            tsplp::OptimizationMode::Max, paths, weights, tsplp::DependencyGraph { weights },
            std::chrono::steady_clock::now() + 1h);
        // expected improvements
        // { { 0, 1, 3, 2, 0 } } => 2 + 8 + 2 + 4
        // { { 0, 3, 1, 2, 0 } } => 4 + 1 + 6 + 4
        // { { 0, 3, 2, 1, 0 } } => 4 + 2 + 5 + 2
        CHECK(improvement == 3);
        CHECK(tsplp::CalculateObjective(tsplp::OptimizationMode::Max, optPaths, weights) == 13);
    }
}

TEST_CASE("twoopt A==2", "[Heuristics]")
{
    // clang-format off
    xt::xarray<double> weights =
    {
        { 0, 2, 3, 4, 0, 2, 3, 4 },
        { 2, 0, 6, 8, 2, 0, 6, 8 },
        { 4, 5, 0, 7, 4, 5, 0, 7 },
        { 0, 1, 2, 0, 0, 1, 2, 0 },
        { 0, 2, 3, 4, 0, 2, 3, 4 },
        { 2, 0, 6, 8, 2, 0, 6, 8 },
        { 4, 5, 0, 7, 4, 5, 0, 7 },
        { 0, 1, 2, 0, 0, 1, 2, 0 }
    };
    // clang-format on

    const std::vector<std::vector<size_t>> paths = { { 0, 1, 3, 2, 0 }, { 5, 7, 6, 4, 5 } };

    {
        const auto [optPaths, improvement] = tsplp::TwoOptPaths(
            tsplp::OptimizationMode::Sum, paths, weights, tsplp::DependencyGraph { weights },
            std::chrono::steady_clock::now() + 1h);
        // expected improvements
        // { { 0, 1, 3, 2, 0 }, { 5, 7, 6, 4, 5 } } => 2 + 8 + 2 + 4, 8 + 2 + 4 + 2 => 16 + 16
        // { { 0, 3, 1, 2, 0 }, { 5, 7, 6, 4, 5 } } => 4 + 1 + 6 + 4, 8 + 2 + 4 + 2 => 15 + 16
        // { { 0, 3, 2, 1, 0 }, { 5, 7, 6, 4, 5 } } => 4 + 2 + 5 + 2, 8 + 2 + 4 + 2 => 13 + 16
        // { { 0, 6, 2, 1, 0 }, { 5, 7, 3, 4, 5 } } => 3 + 0 + 5 + 2, 8 + 0 + 0 + 2 => 10 + 10
        // { { 0, 6, 2, 4, 0 }, { 5, 7, 3, 1, 5 } } => 3 + 0 + 4 + 0, 8 + 0 + 1 + 0 => 7 + 9
        CHECK(improvement == 16);
        CHECK(tsplp::CalculateObjective(tsplp::OptimizationMode::Sum, optPaths, weights) == 16);
    }
    {
        const auto [optPaths, improvement] = tsplp::TwoOptPaths(
            tsplp::OptimizationMode::Max, paths, weights, tsplp::DependencyGraph { weights },
            std::chrono::steady_clock::now() + 1h);
        // expected improvements
        // { { 0, 1, 3, 2, 0 }, { 5, 7, 6, 4, 5 } } => 2 + 8 + 2 + 4, 8 + 2 + 4 + 2 => 16, 16
        // { { 0, 7, 3, 2, 0 }, { 5, 1, 6, 4, 5 } } => 4 + 0 + 2 + 4, 0 + 6 + 4 + 2 => 10, 12
        // { { 0, 7, 6, 2, 0 }, { 5, 1, 3, 4, 5 } } => 4 + 2 + 0 + 4, 0 + 8 + 0 + 2 => 10, 10
        // { { 0, 4, 6, 2, 0 }, { 5, 1, 3, 7, 5 } } => 0 + 3 + 0 + 4, 0 + 8 + 0 + 1 => 7, 9
        CHECK(improvement == 7);
        CHECK(tsplp::CalculateObjective(tsplp::OptimizationMode::Max, optPaths, weights) == 9);
    }
}

TEST_CASE("twoopt A==1 with dependencies", "[Heuristics]")
{
    // 1 -> 3
    // clang-format off
    xt::xarray<double> weights =
    {
        { 0, 2, 3, 4 },
        { 2, 0, 6, 8 },
        { 4, 5, 0, 7 },
        { 0,-1, 2, 0 }
    };
    // clang-format on

    const std::vector<std::vector<size_t>> paths = { { 0, 1, 3, 2, 0 } };

    {
        const auto [optPaths, improvement] = tsplp::TwoOptPaths(
            tsplp::OptimizationMode::Sum, paths, weights, tsplp::DependencyGraph { weights },
            std::chrono::steady_clock::now() + 1h);
        // expected improvements
        // { { 0, 1, 3, 2, 0 } } => 2 + 8 + 2 + 4
        // { { 0, 1, 2, 3, 0 } } => 2 + 6 + 7 + 0
        CHECK(improvement == 1);
        CHECK(tsplp::CalculateObjective(tsplp::OptimizationMode::Sum, optPaths, weights) == 15);
    }
    {
        const auto [optPaths, improvement] = tsplp::TwoOptPaths(
            tsplp::OptimizationMode::Max, paths, weights, tsplp::DependencyGraph { weights },
            std::chrono::steady_clock::now() + 1h);
        // expected improvements
        // { { 0, 1, 3, 2, 0 } } => 2 + 8 + 2 + 4
        // { { 0, 1, 2, 3, 0 } } => 2 + 6 + 7 + 0
        CHECK(improvement == 1);
        CHECK(tsplp::CalculateObjective(tsplp::OptimizationMode::Max, optPaths, weights) == 15);
    }
}

TEST_CASE("twoopt A==2 with dependencies", "[Heuristics]")
{
    // 1->3, 7->6
    // clang-format off
    xt::xarray<double> weights =
    {
        { 0, 2, 3, 4, 0, 2, 3, 4 },
        { 2, 0, 6, 8, 2, 0, 6, 8 },
        { 4, 5, 0, 7, 4, 5, 0, 7 },
        { 0,-1, 2, 0, 0, 1, 2, 0 },
        { 0, 2, 3, 4, 0, 2, 3, 4 },
        { 2, 0, 6, 8, 2, 0, 6, 8 },
        { 4, 5, 0, 7, 4, 5, 0,-1 },
        { 0, 1, 2, 0, 0, 1, 2, 0 }
    };
    // clang-format on

    const std::vector<std::vector<size_t>> paths = { { 0, 1, 3, 2, 0 }, { 5, 7, 6, 4, 5 } };

    {
        const auto [optPaths, improvement] = tsplp::TwoOptPaths(
            tsplp::OptimizationMode::Sum, paths, weights, tsplp::DependencyGraph { weights },
            std::chrono::steady_clock::now() + 1h);
        // expected improvements
        // { { 0, 1, 3, 2, 0 }, { 5, 7, 6, 4, 5 } } => 2 + 8 + 2 + 4, 8 + 2 + 4 + 2 => 16 + 16
        // { { 0, 1, 2, 3, 0 }, { 5, 7, 6, 4, 5 } } => 2 + 6 + 7 + 0, 8 + 2 + 4 + 2 => 15 + 16
        // { { 0, 1, 4, 3, 0 }, { 5, 7, 6, 2, 5 } } => 2 + 2 + 4 + 0, 8 + 2 + 0 + 5 =>  8 + 15
        CHECK(improvement == 9);
        CHECK(tsplp::CalculateObjective(tsplp::OptimizationMode::Sum, optPaths, weights) == 23);
    }
    {
        const auto [optPaths, improvement] = tsplp::TwoOptPaths(
            tsplp::OptimizationMode::Max, paths, weights, tsplp::DependencyGraph { weights },
            std::chrono::steady_clock::now() + 1h);
        // expected improvements
        // { { 0, 1, 3, 2, 0 }, { 5, 7, 6, 4, 5 } } => 2 + 8 + 2 + 4, 8 + 2 + 4 + 2 => 16, 16
        // { { 0, 1, 3, 4, 0 }, { 5, 7, 6, 2, 5 } } => 2 + 8 + 0 + 0, 8 + 2 + 0 + 5 => 10, 15
        CHECK(improvement == 1);
        CHECK(tsplp::CalculateObjective(tsplp::OptimizationMode::Max, optPaths, weights) == 15);
    }
}

TEST_CASE("nearest insertion A==1", "[Heuristics]")
{
    // clang-format off
    xt::xarray<double> weights =
    {
        { 0, 2, 3, 4 },
        { 2, 0, 6, 8 },
        { 4, 5, 0, 7 },
        { 0, 1, 2, 0 }
    };
    // clang-format on

    const xt::xtensor<size_t, 1> startPositions = { 3 };
    const xt::xtensor<size_t, 1> endPositions = { 0 };
    const std::vector<std::vector<size_t>> expectedPaths = { { 3, 2, 1, 0 } };

    {
        const auto [paths, objective] = tsplp::NearestInsertion(
            tsplp::OptimizationMode::Sum, weights, startPositions, endPositions,
            tsplp::DependencyGraph { weights }, std::chrono::steady_clock::now() + 1h);
        // expected insertions
        // { { 3, 0 } } => 0
        // { { 3, 1, 0 } } => 1 + 2
        // { { 3, 2, 1, 0 } } => 2 + 5 + 2
        CHECK(objective == 9);
        CHECK(paths == expectedPaths);
    }
    {
        const auto [paths, objective] = tsplp::NearestInsertion(
            tsplp::OptimizationMode::Max, weights, startPositions, endPositions,
            tsplp::DependencyGraph { weights }, std::chrono::steady_clock::now() + 1h);
        // expected insertions
        // { { 3, 0 } } => 0
        // { { 3, 1, 0 } } => 1 + 2
        // { { 3, 2, 1, 0 } } => 2 + 5 + 2
        CHECK(objective == 9);
        CHECK(paths == expectedPaths);
    }
}

TEST_CASE("nearest insertion A==2", "[Heuristics]")
{
    // clang-format off
    xt::xarray<double> weights =
    {
        { 0, 2, 3, 4, 0, 2, 3, 4 },
        { 2, 0, 6, 8, 2, 0, 6, 8 },
        { 4, 5, 0, 7, 4, 5, 0, 7 },
        { 0, 1, 2, 0, 0, 1, 2, 0 },
        { 0, 2, 3, 4, 0, 2, 3, 4 },
        { 2, 0, 6, 8, 2, 0, 6, 8 },
        { 4, 5, 0, 7, 4, 5, 0, 7 },
        { 0, 1, 2, 0, 0, 1, 2, 0 }
    };
    // clang-format on

    const xt::xtensor<size_t, 1> startPositions = { 7, 3 };
    const xt::xtensor<size_t, 1> endPositions = { 4, 0 };

    {
        const auto [paths, objective] = tsplp::NearestInsertion(
            tsplp::OptimizationMode::Sum, weights, startPositions, endPositions,
            tsplp::DependencyGraph { weights }, std::chrono::steady_clock::now() + 1h);
        // expected insertions
        // { { 7, 4 }, { 3, 0 } } => 0, 0
        // { { 7, 6, 4 }, { 3, 0 } } => 2 + 4, 0
        // { { 7, 6, 5, 4 }, { 3, 0 } } => 2 + 5 + 2, 0
        // { { 7, 2, 6, 5, 4 }, { 3, 0 } } => 2 + 0 + 5 + 2, 0
        // { { 7, 2, 6, 1, 5, 4 }, { 3, 0 } } => 2 + 0 + 5 + 0 + 2, 0
        const std::vector<std::vector<size_t>> expectedPaths = { { 7, 2, 6, 1, 5, 4 }, { 3, 0 } };
        CHECK(objective == 9);
        CHECK(paths == expectedPaths);
    }
    {
        const auto [paths, objective] = tsplp::NearestInsertion(
            tsplp::OptimizationMode::Max, weights, startPositions, endPositions,
            tsplp::DependencyGraph { weights }, std::chrono::steady_clock::now() + 1h);
        // expected insertions
        // { { 7, 4 }, { 3, 0 } } => 0, 0
        // { { 7, 6, 4 }, { 3, 0 } } => 2 + 4, 0
        // { { 7, 6, 4 }, { 3, 5, 0 } } = > 2 + 4, 1 + 2
        // { { 7, 2, 6, 4 }, { 3, 5, 0 } } = > 2 + 0 + 4, 1 + 2
        // { { 7, 2, 6, 4 }, { 3, 1, 5, 0 } } = > 2 + 0 + 4, 1 + 0 + 2
        const std::vector<std::vector<size_t>> expectedPaths = { { 7, 2, 6, 4 }, { 3, 1, 5, 0 } };
        CHECK(objective == 6);
        CHECK(paths == expectedPaths);
    }
}
