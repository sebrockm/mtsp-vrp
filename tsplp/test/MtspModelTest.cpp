#include "MtspModel.hpp"

#include "TsplpExceptions.hpp"

#include <catch2/catch.hpp>

#include <chrono>
#include <thread>

using namespace std::chrono_literals;

constexpr auto timeLimit =
#ifdef NDEBUG
    1s
#else
    10s
#endif
    ;

TEST_CASE("circular start and end", "[MtspModel]")
{
    // clang-format off
    xt::xtensor<int, 2> weights =
    {
        {0, 1, 1},
        {1, 0, 1},
        {1, 2, 0}
    };
    // clang-format on

    xt::xtensor<int, 1> startPositions { 0, 1 };
    xt::xtensor<int, 1> endPositions { 1, 0 };

    tsplp::MtspModel model { startPositions, endPositions, weights, tsplp::OptimizationMode::Sum,
                             timeLimit };
    model.BranchAndCutSolve(1);
    const auto& result = model.GetResult();

    REQUIRE(!result.IsTimeoutHit());
    REQUIRE(result.GetBounds().Lower == 3);
    REQUIRE(result.GetBounds().Upper == 3);
    REQUIRE(
        result.GetPaths()
        == std::vector { std::vector<size_t> { 0, 1 }, std::vector<size_t> { 1, 2, 0 } });
}

TEST_CASE("Timeout", "[MtspModel]")
{
    // clang-format off
    xt::xtensor<int, 2> weights =
    {
        {0, 1, 1},
        {1, 0, 1},
        {1, 2, 0}
    };
    // clang-format on

    xt::xtensor<int, 1> startPositions { 0 };
    xt::xtensor<int, 1> endPositions { 0 };

    tsplp::MtspModel model { startPositions, endPositions, weights, tsplp::OptimizationMode::Sum,
                             100ms };
    std::this_thread::sleep_for(100ms);

    model.BranchAndCutSolve(1);
    const auto& result = model.GetResult();

    REQUIRE(result.IsTimeoutHit());
}
