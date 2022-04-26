#include "MtspModel.hpp"
#include "TsplpExceptions.hpp"

#include <catch2/catch.hpp>
#include <chrono>

#include <xtensor/xnpy.hpp>
#include <fstream>

// TEST_CASE("specific instance", "[instances]")
// {
//     std::ifstream stream("/Users/sebastian/repos/mtsp-vrp/tsplib/tsplib/tsp/linhp318.tsp.weights.npy");
//     const auto weights = xt::load_npy<int>(stream);

//     xt::xtensor<int, 1> startPositions{ 0 };
//     xt::xtensor<int, 1> endPositions{ 0 };

//     tsplp::MtspModel model{ startPositions, endPositions, weights, std::chrono::minutes{ 5 } };
//     auto result = model.BranchAndCutSolve();

//     REQUIRE(result.LowerBound <= Approx(41345));
//     REQUIRE(result.UpperBound >= Approx(41345));
// }

void CheckResult(tsplp::MtspResult const& result, xt::xtensor<size_t, 1> const& sp, xt::xtensor<size_t, 1> const& ep, xt::xtensor<int, 2> const& weights)
{
    const auto N = weights.shape(0);
    const auto A = sp.size();

    double sum = 0;
    std::vector<size_t> visited(N, A);

    for (size_t a = 0; a < A; ++a)
    {
        CHECK(result.Paths[a].front() == sp[a]);
        if (sp[a] != ep[a])
            CHECK(result.Paths[a].back() == ep[a]);
        
        for (size_t i = 0; i + 1 < result.Paths[a].size(); ++i)
        {
            const auto w = weights(result.Paths[a][i], result.Paths[a][i+1]);
            CAPTURE(a, i, result.Paths[a][i], result.Paths[a][i+1]);
            CHECK(w >= 0);

            sum += w;
            
            if (result.Paths[a][i] != sp[a] && result.Paths[a][i] != ep[a])
                CHECK(visited[result.Paths[a][i]] == A);
            visited[result.Paths[a][i]] = a;
        }

        if (sp[a] != ep[a])
        {
            CHECK(visited[result.Paths[a].back()] == A);
            visited[result.Paths[a].back()] = a;
        }
        else
        {
            const auto w = weights(result.Paths[a].back(), result.Paths[a].front());
            CHECK(w >= 0);
            sum += w;
        }
    }

    for (const auto v : visited)
    {
        CHECK(v != A);
    }

    CHECK(result.LowerBound <= Approx(sum));
    CHECK(Approx(sum) <= result.UpperBound);
}

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

    tsplp::MtspModel model{ startPositions, endPositions, weights, std::chrono::seconds{ 1 } };
    auto result = model.BranchAndCutSolve();

    CheckResult(result, startPositions, endPositions, weights);
    
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

    tsplp::MtspModel model{ startPositions, endPositions, weights, std::chrono::seconds{ 1 } };
    auto result = model.BranchAndCutSolve();

    CheckResult(result, startPositions, endPositions, weights);

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
    xt::xtensor<int, 1> endPositions{ 8 };

    tsplp::MtspModel model{ startPositions, endPositions, weights, std::chrono::seconds{ 1 } };
    auto result = model.BranchAndCutSolve();

    CheckResult(result, startPositions, endPositions, weights);

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
    xt::xtensor<int, 1> endPositions{ 8, 8, 8, 8 };

    REQUIRE_THROWS_AS(tsplp::MtspModel(startPositions, endPositions, weights, std::chrono::seconds{ 1 }), tsplp::IncompatibleDependenciesException);
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

    tsplp::MtspModel model{ startPositions, endPositions, weights, std::chrono::minutes{ 1 } };
    auto result = model.BranchAndCutSolve();

    CheckResult(result, startPositions, endPositions, weights);

    REQUIRE(result.LowerBound == Approx(1200));
    REQUIRE(result.UpperBound == Approx(1200));
}
