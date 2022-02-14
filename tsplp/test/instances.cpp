#include "MtspModel.hpp"
#include "TsplpExceptions.hpp"

#include <catch2/catch.hpp>

#include <xtensor/xnpy.hpp>

TEST_CASE("br17.atsp", "[instances]")
{
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

    xt::xtensor<int, 1> startPositions{ 0 };
    xt::xtensor<int, 1> endPositions{ 0 };

    tsplp::MtspModel model{ startPositions, endPositions, weights };
    auto result = model.BranchAndCutSolve(std::chrono::seconds{ 1 });
    
    REQUIRE(result.LowerBound == Approx(39));
    REQUIRE(result.UpperBound == Approx(39));
}

TEST_CASE("br17.atsp 4 agents vrp", "[instances]")
{
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

    xt::xtensor<int, 1> startPositions{ 0, 0, 0, 0 };
    xt::xtensor<int, 1> endPositions{ 0, 0, 0, 0 };

    tsplp::MtspModel model{ startPositions, endPositions, weights };
    auto result = model.BranchAndCutSolve(std::chrono::hours{ 1 });

    REQUIRE(result.LowerBound == Approx(39));
    REQUIRE(result.UpperBound == Approx(39));
}

TEST_CASE("ESC07.sop", "[instances]")
{
    xt::xtensor<int, 2> weights =
    {
        {  0,   0,    0,    0,    0,    0,    0,    0, 1000000 },
        { -1,   0,  100,  200,   75,    0,  300,  100, 0 },
        { -1, 400,    0,  500,  325,  400,  600,    0, 0 },
        { -1, 700,  800,    0,  550,  700,  900,  800, 0 },
        { -1,  -1,  250,  225,    0,  275,  525,  250, 0 },
        { -1,  -1,  100,  200,   -1,    0,   -1,   -1, 0 },
        { -1,  -1, 1100, 1200, 1075, 1000,    0, 1100, 0 },
        { -1,  -1,    0,  500,  325,  400,  600,    0, 0 },
        { -1,  -1,   -1,   -1,   -1,   -1,   -1,   -1, 0 }
    };

    xt::xtensor<int, 1> startPositions{ 0 };
    xt::xtensor<int, 1> endPositions{ 0 };

    tsplp::MtspModel model{ startPositions, endPositions, weights };
    auto result = model.BranchAndCutSolve(std::chrono::seconds{ 1 });

    REQUIRE(result.LowerBound == Approx(2125));
    REQUIRE(result.UpperBound == Approx(2125));
}

TEST_CASE("ESC07.sop 4 agents vrp incompatible", "[instances]")
{
    xt::xtensor<int, 2> weights =
    {
        {  0,   0,    0,    0,    0,    0,    0,    0, 1000000 },
        { -1,   0,  100,  200,   75,    0,  300,  100, 0 },
        { -1, 400,    0,  500,  325,  400,  600,    0, 0 },
        { -1, 700,  800,    0,  550,  700,  900,  800, 0 },
        { -1,  -1,  250,  225,    0,  275,  525,  250, 0 },
        { -1,  -1,  100,  200,   -1,    0,   -1,   -1, 0 },
        { -1,  -1, 1100, 1200, 1075, 1000,    0, 1100, 0 },
        { -1,  -1,    0,  500,  325,  400,  600,    0, 0 },
        { -1,  -1,   -1,   -1,   -1,   -1,   -1,   -1, 0 }
    };

    xt::xtensor<int, 1> startPositions{ 0, 0, 0, 0 };
    xt::xtensor<int, 1> endPositions{ 0, 0, 0, 0 };

    tsplp::MtspModel model{ startPositions, endPositions, weights };
    REQUIRE_THROWS_AS(model.BranchAndCutSolve(std::chrono::seconds{ 1 }), tsplp::IncompatibleDependenciesException);
}

TEST_CASE("ESC07.sop 4 agents vrp", "[instances]")
{
    xt::xtensor<int, 2> weights =
    {
        {   0,   0,    0,    0,    0,    0,    0,    0, 1000000 },
        { 100,   0,  100,  200,   75,    0,  300,  100, 0 },
        { 100, 400,    0,  500,  325,  400,  600,    0, 0 },
        { 100, 700,  800,    0,  550,  700,  900,  800, 0 },
        { 100,  -1,  250,  225,    0,  275,  525,  250, 0 },
        { 100,  -1,  100,  200,   -1,    0,   -1,   -1, 0 },
        { 100,  -1, 1100, 1200, 1075, 1000,    0, 1100, 0 },
        { 100,  -1,    0,  500,  325,  400,  600,    0, 0 },
        { 100, 100,  100,  100,  100,  100,  100,  100, 0 }
    };

    xt::xtensor<int, 1> startPositions{ 0, 0, 0, 0 };
    xt::xtensor<int, 1> endPositions{ 0, 0, 0, 0 };

    tsplp::MtspModel model{ startPositions, endPositions, weights };
    auto result = model.BranchAndCutSolve(std::chrono::seconds{ 1 });

    REQUIRE(result.LowerBound == Approx(1200));
    REQUIRE(result.UpperBound == Approx(1200));
}

//TEST_CASE("configurable bench", "[instances]")
//{
//    const auto weights = xt::load_npy<int>("C:\\Users\\sebas\\source\\repos\\mtsp-vrp\\tsplib\\tsplib\\tsp\\bayg29.tsp.weights.npy");
//
//    xt::xtensor<int, 1> startPositions{ 0 };
//    xt::xtensor<int, 1> endPositions{ 0 };
//
//    tsplp::MtspModel model{ startPositions, endPositions, weights };
//    auto result = model.BranchAndCutSolve(std::chrono::minutes{ 10 });
//
//    REQUIRE(result.LowerBound == Approx(1163));
//    REQUIRE(result.UpperBound == Approx(1163));
//}

//TEST_CASE("configurable bench 4 agents vrp", "[instances]")
//{
//    const auto weights = xt::load_npy<int>("C:\\Users\\sebas\\source\\repos\\mtsp-vrp\\tsplib\\tsplib\\sop\\ft70.1.sop.weights.npy");
//
//    xt::xtensor<int, 1> startPositions{ 0, 0, 0, 0 };
//    xt::xtensor<int, 1> endPositions{ 0, 0, 0, 0 };
//
//    tsplp::MtspModel model{ startPositions, endPositions, weights };
//    auto result = model.BranchAndCutSolve(std::chrono::seconds{ 10 });
//
//    REQUIRE(result.IsTimeoutHit);
//}
