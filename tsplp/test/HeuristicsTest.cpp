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
    REQUIRE(tsplp::CalculatePathLength({ }, weights) == 0);
}
