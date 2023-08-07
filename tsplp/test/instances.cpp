#include "MtspModel.hpp"
#include "TsplpExceptions.hpp"

#include <catch2/catch.hpp>

#include <chrono>

constexpr auto timeLimit = std::chrono::seconds {
#ifdef NDEBUG
    1
#else
    100
#endif
};

TEST_CASE("br17.atsp", "[instances]")
{
    // clang-format off
    xt::xtensor<int, 2> weights =
    {
        {9999, 3, 5, 48, 48, 8, 8, 5, 5, 3, 3, 0, 3, 5, 8, 8, 5},
        {3, 9999, 3, 48, 48, 8, 8, 5, 5, 0, 0, 3, 0, 3, 8, 8, 5},
        {5, 3, 9999, 72, 72, 48, 48, 24, 24, 3, 3, 5, 3, 0, 48, 48, 24},
        {48,48, 74, 9999, 0, 6, 6, 12, 12, 48, 48, 48, 48, 74, 6, 6, 12},
        {48,48, 74, 0, 9999, 6, 6, 12, 12, 48, 48, 48, 48, 74, 6, 6, 12},
        {8, 8, 50, 6, 6, 9999, 0, 8, 8, 8, 8, 8, 8, 50, 0, 0, 8},
        {8, 8, 50, 6, 6, 0, 9999, 8, 8, 8, 8, 8, 8, 50, 0, 0, 8},
        {5, 5, 26, 12, 12, 8, 8, 9999, 0, 5, 5, 5, 5, 26, 8, 8, 0},
        {5, 5, 26, 12, 12, 8, 8, 0, 9999, 5, 5, 5, 5, 26, 8, 8, 0},
        {3, 0, 3, 48, 48, 8, 8, 5, 5, 9999, 0, 3, 0, 3, 8, 8, 5},
        {3, 0, 3, 48, 48, 8, 8, 5, 5, 0, 9999, 3, 0, 3, 8, 8, 5},
        {0, 3, 5, 48, 48, 8, 8, 5, 5, 3, 3, 9999, 3, 5, 8, 8, 5},
        {3, 0, 3, 48, 48, 8, 8, 5, 5, 0, 0, 3, 9999, 3, 8, 8, 5},
        {5, 3, 0, 72, 72, 48, 48, 24, 24, 3, 3, 5, 3, 9999, 48, 48, 24},
        {8, 8, 50, 6, 6, 0, 0, 8, 8, 8, 8, 8, 8, 50, 9999, 0, 8},
        {8, 8, 50, 6, 6, 0, 0, 8, 8, 8, 8, 8, 8, 50, 0, 9999, 8},
        {5, 5, 26, 12, 12, 8, 8, 0, 0, 5, 5, 5, 5, 26, 8, 8, 9999}
    };
    // clang-format on

    xt::xtensor<int, 1> startPositions { 0 };
    xt::xtensor<int, 1> endPositions { 0 };

    const auto startTime = std::chrono::steady_clock::now();

    tsplp::MtspModel model { startPositions, endPositions, weights, tsplp::OptimizationMode::Sum,
                             timeLimit };
    model.BranchAndCutSolve();
    const auto& result = model.GetResult();

    const auto endTime = std::chrono::steady_clock::now();
    CAPTURE(result.GetPaths());
    CHECK(endTime - startTime <= timeLimit);
    CHECK(!result.IsTimeoutHit());
    CHECK(result.GetLowerBound() == Approx(39));
    CHECK(result.GetUpperBound() == Approx(39));
}

TEST_CASE("br17.atsp 4 agents vrp", "[instances]")
{
    // clang-format off
    xt::xtensor<int, 2> weights =
    {
        {9999, 3, 5, 48, 48, 8, 8, 5, 5, 3, 3, 0, 3, 5, 8, 8, 5},
        {3, 9999, 3, 48, 48, 8, 8, 5, 5, 0, 0, 3, 0, 3, 8, 8, 5},
        {5, 3, 9999, 72, 72, 48, 48, 24, 24, 3, 3, 5, 3, 0, 48, 48, 24},
        {48,48, 74, 9999, 0, 6, 6, 12, 12, 48, 48, 48, 48, 74, 6, 6, 12},
        {48,48, 74, 0, 9999, 6, 6, 12, 12, 48, 48, 48, 48, 74, 6, 6, 12},
        {8, 8, 50, 6, 6, 9999, 0, 8, 8, 8, 8, 8, 8, 50, 0, 0, 8},
        {8, 8, 50, 6, 6, 0, 9999, 8, 8, 8, 8, 8, 8, 50, 0, 0, 8},
        {5, 5, 26, 12, 12, 8, 8, 9999, 0, 5, 5, 5, 5, 26, 8, 8, 0},
        {5, 5, 26, 12, 12, 8, 8, 0, 9999, 5, 5, 5, 5, 26, 8, 8, 0},
        {3, 0, 3, 48, 48, 8, 8, 5, 5, 9999, 0, 3, 0, 3, 8, 8, 5},
        {3, 0, 3, 48, 48, 8, 8, 5, 5, 0, 9999, 3, 0, 3, 8, 8, 5},
        {0, 3, 5, 48, 48, 8, 8, 5, 5, 3, 3, 9999, 3, 5, 8, 8, 5},
        {3, 0, 3, 48, 48, 8, 8, 5, 5, 0, 0, 3, 9999, 3, 8, 8, 5},
        {5, 3, 0, 72, 72, 48, 48, 24, 24, 3, 3, 5, 3, 9999, 48, 48, 24},
        {8, 8, 50, 6, 6, 0, 0, 8, 8, 8, 8, 8, 8, 50, 9999, 0, 8},
        {8, 8, 50, 6, 6, 0, 0, 8, 8, 8, 8, 8, 8, 50, 0, 9999, 8},
        {5, 5, 26, 12, 12, 8, 8, 0, 0, 5, 5, 5, 5, 26, 8, 8, 9999}
    };
    // clang-format on

    xt::xtensor<int, 1> startPositions { 0, 0, 0, 0 };
    xt::xtensor<int, 1> endPositions { 0, 0, 0, 0 };

    const auto startTime = std::chrono::steady_clock::now();

    tsplp::MtspModel model { startPositions, endPositions, weights, tsplp::OptimizationMode::Sum,
                             timeLimit };
    model.BranchAndCutSolve();
    const auto& result = model.GetResult();

    const auto endTime = std::chrono::steady_clock::now();
    CAPTURE(result.GetPaths());
    CHECK(endTime - startTime <= timeLimit);
    CHECK(!result.IsTimeoutHit());
    CHECK(result.GetLowerBound() == Approx(39));
    CHECK(result.GetUpperBound() == Approx(39));
}

// TEST_CASE("ESC07.sop", "[instances]")
// {
//     // clang-format off
//     xt::xtensor<int, 2> weights =
//     {
//         {  0,   0,    0,    0,    0,    0,    0,    0, 1000000 },
//         { -1,   0,  100,  200,   75,    0,  300,  100, 0 },
//         { -1, 400,    0,  500,  325,  400,  600,    0, 0 },
//         { -1, 700,  800,    0,  550,  700,  900,  800, 0 },
//         { -1,  -1,  250,  225,    0,  275,  525,  250, 0 },
//         { -1,  -1,  100,  200,   -1,    0,   -1,   -1, 0 },
//         { -1,  -1, 1100, 1200, 1075, 1000,    0, 1100, 0 },
//         { -1,  -1,    0,  500,  325,  400,  600,    0, 0 },
//         { -1,  -1,   -1,   -1,   -1,   -1,   -1,   -1, 0 }
//     };
//     // clang-format on

//     xt::xtensor<int, 1> startPositions { 0 };
//     xt::xtensor<int, 1> endPositions { 8 };

//     const auto startTime = std::chrono::steady_clock::now();

//     tsplp::MtspModel model { startPositions, endPositions, weights, tsplp::OptimizationMode::Sum,
//                              timeLimit };
//     model.BranchAndCutSolve();
//     const auto& result = model.GetResult();

//     const auto endTime = std::chrono::steady_clock::now();
//     CHECK(endTime - startTime <= timeLimit);
//     CHECK(!result.IsTimeoutHit());
//     CHECK(result.GetLowerBound() == Approx(2125));
//     CHECK(result.GetUpperBound() == Approx(2125));
//     CAPTURE(result.GetPaths());
// }

// TEST_CASE("ESC07.sop start end same", "[instances]")
// {
//     // clang-format off
//     xt::xtensor<int, 2> weights =
//     {
//         {  0,   0,    0,    0,    0,    0,    0,    0, 1000000 },
//         { -1,   0,  100,  200,   75,    0,  300,  100, 0 },
//         { -1, 400,    0,  500,  325,  400,  600,    0, 0 },
//         { -1, 700,  800,    0,  550,  700,  900,  800, 0 },
//         { -1,  -1,  250,  225,    0,  275,  525,  250, 0 },
//         { -1,  -1,  100,  200,   -1,    0,   -1,   -1, 0 },
//         { -1,  -1, 1100, 1200, 1075, 1000,    0, 1100, 0 },
//         { -1,  -1,    0,  500,  325,  400,  600,    0, 0 },
//         { -1,  -1,   -1,   -1,   -1,   -1,   -1,   -1, 0 }
//     };
//     // clang-format on

//     xt::xtensor<int, 1> startPositions { 0 };
//     xt::xtensor<int, 1> endPositions { 0 };

//     const auto startTime = std::chrono::steady_clock::now();

//     tsplp::MtspModel model { startPositions, endPositions, weights, tsplp::OptimizationMode::Sum,
//                              timeLimit };
//     model.BranchAndCutSolve();
//     const auto& result = model.GetResult();

//     const auto endTime = std::chrono::steady_clock::now();
//     CHECK(endTime - startTime <= timeLimit);

//     CHECK(!result.IsTimeoutHit());
//     CHECK(result.GetLowerBound() == Approx(2125));
//     CHECK(result.GetUpperBound() == Approx(2125));
//     CAPTURE(result.GetPaths());
// }

// TEST_CASE("ESC07.sop 4 agents vrp incompatible", "[instances]")
// {
//     // clang-format off
//     xt::xtensor<int, 2> weights =
//     {
//         {  0,   0,    0,    0,    0,    0,    0,    0, 1000000 },
//         { -1,   0,  100,  200,   75,    0,  300,  100, 0 },
//         { -1, 400,    0,  500,  325,  400,  600,    0, 0 },
//         { -1, 700,  800,    0,  550,  700,  900,  800, 0 },
//         { -1,  -1,  250,  225,    0,  275,  525,  250, 0 },
//         { -1,  -1,  100,  200,   -1,    0,   -1,   -1, 0 },
//         { -1,  -1, 1100, 1200, 1075, 1000,    0, 1100, 0 },
//         { -1,  -1,    0,  500,  325,  400,  600,    0, 0 },
//         { -1,  -1,   -1,   -1,   -1,   -1,   -1,   -1, 0 }
//     };
//     // clang-format on

//     xt::xtensor<int, 1> startPositions { 0, 0, 0, 0 };
//     xt::xtensor<int, 1> endPositions { 0, 0, 0, 0 };

//     CHECK_THROWS_AS(
//         tsplp::MtspModel(
//             startPositions, endPositions, weights, tsplp::OptimizationMode::Sum, timeLimit),
//         tsplp::IncompatibleDependenciesException);
// }

// TEST_CASE("ESC07.sop 4 agents vrp", "[instances]")
// {
//     // clang-format off
//     xt::xtensor<int, 2> weights =
//     {
//         {   0,   0,    0,    0,    0,    0,    0,    0, 1000000 },
//         { 100,   0,  100,  200,   75,    0,  300,  100, 0 },
//         { 100, 400,    0,  500,  325,  400,  600,    0, 0 },
//         { 100, 700,  800,    0,  550,  700,  900,  800, 0 },
//         { 100,  -1,  250,  225,    0,  275,  525,  250, 0 },
//         { 100,  -1,  100,  200,   -1,    0,   -1,   -1, 0 },
//         { 100,  -1, 1100, 1200, 1075, 1000,    0, 1100, 0 },
//         { 100,  -1,    0,  500,  325,  400,  600,    0, 0 },
//         { 100, 100,  100,  100,  100,  100,  100,  100, 0 }
//     };
//     // clang-format on

//     xt::xtensor<int, 1> startPositions { 0, 0, 0, 0 };
//     xt::xtensor<int, 1> endPositions { 0, 0, 0, 0 };

//     const auto startTime = std::chrono::steady_clock::now();

//     tsplp::MtspModel model { startPositions, endPositions, weights, tsplp::OptimizationMode::Sum,
//                              timeLimit };
//     model.BranchAndCutSolve();
//     const auto& result = model.GetResult();

//     const auto endTime = std::chrono::steady_clock::now();
//     CHECK(endTime - startTime <= timeLimit);
//     CHECK(!result.IsTimeoutHit());
//     CHECK(result.GetLowerBound() == Approx(1200));
//     CHECK(result.GetUpperBound() == Approx(1200));
//     CAPTURE(result.GetPaths());
// }
