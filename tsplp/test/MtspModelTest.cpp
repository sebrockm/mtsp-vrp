#include "MtspModel.hpp"

#include "TsplpExceptions.hpp"

#include <catch2/catch.hpp>

#include <chrono>

constexpr auto timeLimit = std::chrono::seconds {
#ifdef NDEBUG
    1
#else
    10
#endif
};

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
    REQUIRE(result.GetLowerBound() == 3);
    REQUIRE(result.GetUpperBound() == 3);
    REQUIRE(
        result.GetPaths()
        == std::vector { std::vector<size_t> { 0, 1 }, std::vector<size_t> { 1, 2, 0 } });
}
