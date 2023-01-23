#include "WeightManager.hpp"

#include <catch2/catch.hpp>

TEST_CASE("wrong input", "[WeightManager]")
{
    REQUIRE_THROWS([&] { tsplp::WeightManager { { { 0, 1, 2 }, { 3, 4, 5 } }, { 0 }, { 0 } }; }());
    REQUIRE_THROWS([&] { tsplp::WeightManager { { { 0 } }, { 0 }, { 0 } }; }());
    REQUIRE_THROWS([&] { tsplp::WeightManager { { { 0, 1 }, { 2, 3 } }, { 0 }, { 0, 1 } }; }());
    REQUIRE_THROWS([&] { tsplp::WeightManager { { { 0, 1 }, { 2, 3 } }, {}, {} }; }());
    REQUIRE_THROWS([&] { tsplp::WeightManager { { { 0, 1 }, { 2, 3 } }, { 4 }, { 1 } }; }());
    REQUIRE_THROWS([&] { tsplp::WeightManager { { { 0, 1 }, { 2, 3 } }, { 1, 0 }, { 1, 2 } }; }());
}

TEST_CASE("no changes", "[WeightManager]")
{
    // clang-format off
    const xt::xtensor<int, 2> weights = 
    { 
        { 0, 1, 2, 3, 4, 5 },
        { 1, 0, 0, 4, 5, 6 },
        { 2, 3, 0, 5, 6, 7 },
        { 0, 4, 5, 0, 7, 8 },
        { 4, 5, 6, 7, 0, 9 },
        { 5, 6, 7, 8, 9, 0 }
    };
    // clang-format on

    const xt::xtensor<size_t, 1> sp = { 0, 2 };
    const xt::xtensor<size_t, 1> ep = { 1, 3 };
    const tsplp::WeightManager wm { weights, sp, ep };

    REQUIRE(wm.A() == 2);
    REQUIRE(wm.N() == 6);

    REQUIRE(wm.W() == weights);
    REQUIRE(wm.StartPositions() == sp);
    REQUIRE(wm.EndPositions() == ep);

    const std::vector<std::vector<size_t>> paths = { { 0, 4, 1 }, { 2, 5, 3 } };

    REQUIRE(wm.TransformPathsBack(paths) == paths);

    const auto tensor = xt::stack(xt::xtuple(0.1 * weights, 0.2 * weights));
    const auto backtransformedTensor = wm.TransformTensorBack(tensor);

    REQUIRE(
        std::vector(backtransformedTensor.begin(), backtransformedTensor.end())
        == std::vector(tensor.begin(), tensor.end()));
}

TEST_CASE("vrp", "[WeightManager]")
{
    const xt::xtensor<int, 2> weights = { { 0, 1 }, { 2, 0 } }; // W(0, 1) == 1, W(1, 0) == 2
    const xt::xtensor<size_t, 1> sp = { 0, 0, 0, 0 };
    const xt::xtensor<size_t, 1> ep = { 0, 0, 0, 0 };
    const tsplp::WeightManager wm { weights, sp, ep };

    REQUIRE(wm.A() == 4);
    REQUIRE(wm.N() == 9);

    // clang-format off
    const xt::xtensor<int, 2> expectedWeights =
    {
        { 0, 1, 0, 0, 0, 0, 0, 0, 0 },
        { 2, 0, 2, 2, 2, 2, 2, 2, 2 },
        { 0, 1, 0, 0, 0, 0, 0, 0, 0 },
        { 0, 1, 0, 0, 0, 0, 0, 0, 0 },
        { 0, 1, 0, 0, 0, 0, 0, 0, 0 },
        { 0, 1, 0, 0, 0, 0, 0, 0, 0 },
        { 0, 1, 0, 0, 0, 0, 0, 0, 0 },
        { 0, 1, 0, 0, 0, 0, 0, 0, 0 },
        { 0, 1, 0, 0, 0, 0, 0, 0, 0 }
    };
    // clang-format on

    REQUIRE(wm.W() == expectedWeights);

    const auto& spt = wm.StartPositions();
    const auto& ept = wm.EndPositions();
    const xt::xtensor<size_t, 1> pt
        = xt::concatenate(xt::xtuple(wm.StartPositions(), wm.EndPositions()));
    const std::vector<size_t> expectedPt = { 0, 2, 3, 4, 5, 6, 7, 8 };

    REQUIRE_THAT(std::vector(pt.begin(), pt.end()), Catch::Matchers::UnorderedEquals(expectedPt));

    const auto transformedPaths = std::vector<std::vector<size_t>> {
        { spt(0), 1, ept(0) }, { spt(1), ept(1) }, { spt(2), ept(2) }, { spt(3), ept(3) }
    };
    const auto expectedBacktransformedPath
        = std::vector<std::vector<size_t>> { { 0, 1, 0 }, { 0, 0 }, { 0, 0 }, { 0, 0 } };

    REQUIRE(wm.TransformPathsBack(transformedPaths) == expectedBacktransformedPath);

    constexpr double V = 1. / 64;
    constexpr double W = 1. / 8;

    // clang-format off
    const xt::xtensor<double, 3> transformedTensor =
    {
        {
            { 0, V, 0, 0, 0, 0, 0, 0, 0 },
            { V, 0, V, V, V, V, V, V, V },
            { 0, V, 0, 0, 0, 0, 0, 0, 0 },
            { 0, V, 0, 0, 0, 0, 0, 0, 0 },
            { 0, V, 0, 0, 0, 0, 0, 0, 0 },
            { 0, V, 0, 0, 0, 0, 0, 0, 0 },
            { 0, V, 0, 0, 0, 0, 0, 0, 0 },
            { 0, V, 0, 0, 0, 0, 0, 0, 0 },
            { 0, V, 0, 0, 0, 0, 0, 0, 0 }
        },
        {
            { 0, V, 0, 0, 0, 0, 0, 0, 0 },
            { V, 0, V, V, V, V, V, V, V },
            { 0, V, 0, 0, 0, 0, 0, 0, 0 },
            { 0, V, 0, 0, 0, 0, 0, 0, 0 },
            { 0, V, 0, 0, 0, 0, 0, 0, 0 },
            { 0, V, 0, 0, 0, 0, 0, 0, 0 },
            { 0, V, 0, 0, 0, 0, 0, 0, 0 },
            { 0, V, 0, 0, 0, 0, 0, 0, 0 },
            { 0, V, 0, 0, 0, 0, 0, 0, 0 }
        },
        {
            { 0, V, 0, 0, 0, 0, 0, 0, 0 },
            { V, 0, V, V, V, V, V, V, V },
            { 0, V, 0, 0, 0, 0, 0, 0, 0 },
            { 0, V, 0, 0, 0, 0, 0, 0, 0 },
            { 0, V, 0, 0, 0, 0, 0, 0, 0 },
            { 0, V, 0, 0, 0, 0, 0, 0, 0 },
            { 0, V, 0, 0, 0, 0, 0, 0, 0 },
            { 0, V, 0, 0, 0, 0, 0, 0, 0 },
            { 0, V, 0, 0, 0, 0, 0, 0, 0 }
        },
        {
            { 0, V, 0, 0, 0, 0, 0, 0, 0 },
            { V, 0, V, V, V, V, V, V, V },
            { 0, V, 0, 0, 0, 0, 0, 0, 0 },
            { 0, V, 0, 0, 0, 0, 0, 0, 0 },
            { 0, V, 0, 0, 0, 0, 0, 0, 0 },
            { 0, V, 0, 0, 0, 0, 0, 0, 0 },
            { 0, V, 0, 0, 0, 0, 0, 0, 0 },
            { 0, V, 0, 0, 0, 0, 0, 0, 0 },
            { 0, V, 0, 0, 0, 0, 0, 0, 0 }
        }
    };

    const xt::xtensor<double, 3> expectedBacktransformedTensor =
    {
        {
            { 0, W },
            { W, 0 }
        },
        {
            { 0, W },
            { W, 0 }
        },
        {
            { 0, W },
            { W, 0 }
        },
        {
            { 0, W },
            { W, 0 }
        }
    };
    // clang-format on

    const auto backtransformedTensor = wm.TransformTensorBack(transformedTensor);
    REQUIRE(
        std::vector(backtransformedTensor.begin(), backtransformedTensor.end())
        == std::vector(expectedBacktransformedTensor.begin(), expectedBacktransformedTensor.end()));
}

TEST_CASE("start-end-circle", "[WeightManager]")
{
    // clang-format off
    const xt::xtensor<int, 2> weights = 
    {
        { 0, 1, 2 },
        { 3, 0, 4 },
        { 5, 6, 0 }
    };
    // clang format on

    const xt::xtensor<size_t, 1> sp = { 0, 2 };
    const xt::xtensor<size_t, 1> ep = { 2, 0 };
    const tsplp::WeightManager wm { weights, sp, ep };

    REQUIRE(wm.A() == 2);
    REQUIRE(wm.N() == 5);

    // clang-format off
    const xt::xtensor<int, 2> expectedWeights =
    {
        { 0, 1, 2, 2, 0 },
        { 3, 0, 4, 4, 3 },
        { 5, 6, 0, 0, 5 },
        { 5, 6, 0, 0, 5 },
        { 0, 1, 2, 2, 0 }
    };
    // clang-format on

    REQUIRE(wm.W() == expectedWeights);

    const auto& spt = wm.StartPositions();
    const auto& ept = wm.EndPositions();
    const xt::xtensor<size_t, 1> pt
        = xt::concatenate(xt::xtuple(wm.StartPositions(), wm.EndPositions()));
    const std::vector<size_t> expectedPt = { 0, 2, 3, 4 };

    REQUIRE_THAT(std::vector(pt.begin(), pt.end()), Catch::Matchers::UnorderedEquals(expectedPt));

    const auto transformedPaths
        = std::vector<std::vector<size_t>> { { spt(0), 1, ept(0) }, { spt(1), ept(1) } };
    const auto expectedBacktransformedPath
        = std::vector<std::vector<size_t>> { { 0, 1, 2 }, { 2, 0 } };

    REQUIRE(wm.TransformPathsBack(transformedPaths) == expectedBacktransformedPath);

    // clang-format off
    const xt::xtensor<double, 3> transformedTensor =
    {
        {
            { 0, 1, 0, 0, 0 },
            { 0, 0, 1, 0, 0 },
            { 0, 0, 0, 0, 0 },
            { 0, 0, 0, 0, 0 },
            { 0, 0, 0, 0, 0 }
        },
        {
            { 0, 0, 0, 0, 0 },
            { 0, 0, 0, 0, 0 },
            { 0, 0, 0, 0, 0 },
            { 0, 0, 0, 0, 1 },
            { 0, 0, 0, 0, 0 }
        }
    };

    const xt::xtensor<double, 3> expectedBacktransformedTensor =
    {
        {
            { 0, 1, 0 },
            { 0, 0, 1 },
            { 0, 0, 0 }
        },
        {
            { 0, 0, 0 },
            { 0, 0, 0 },
            { 1, 0, 0 }
        }
    };
    // clang-format on

    const auto backtransformedTensor = wm.TransformTensorBack(transformedTensor);
    REQUIRE(
        std::vector(backtransformedTensor.begin(), backtransformedTensor.end())
        == std::vector(expectedBacktransformedTensor.begin(), expectedBacktransformedTensor.end()));
}
