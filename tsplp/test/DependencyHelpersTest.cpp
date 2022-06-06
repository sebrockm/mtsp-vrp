#include "DependencyHelpers.hpp"

#include "MtspModel.hpp"
#include "TsplpExceptions.hpp"

#include <catch2/catch.hpp>

TEST_CASE("empty", "[CreateTransitiveDependencies]")
{
    const auto w = tsplp::CreateTransitiveDependencies({});

    REQUIRE(w.shape(0) == 0);
    REQUIRE(w.shape(1) == 0);
}

TEST_CASE("single", "[CreateTransitiveDependencies]")
{
    const auto w = xt::xtensor<int, 2> { { 0, 0 }, { -1, 0 } }; // 0->1
    const auto wt = tsplp::CreateTransitiveDependencies(w);

    REQUIRE(w == wt);
}

TEST_CASE("single not first", "[CreateTransitiveDependencies]")
{
    xt::xtensor<int, 2> w = xt::zeros<int>({ 200, 200 });
    w(100, 17) = -1;
    const auto wt = tsplp::CreateTransitiveDependencies(w);

    REQUIRE(std::as_const(w) == wt);
}

TEST_CASE("transitive line", "[CreateTransitiveDependencies]")
{
    xt::xtensor<int, 2> w = xt::zeros<int>({ 5, 5 });
    w(1, 0) = -1;
    w(2, 1) = -1;
    w(3, 2) = -1;
    w(4, 3) = -1;
    const auto wt = tsplp::CreateTransitiveDependencies(w);

    w(2, 0) = w(3, 0) = w(4, 0) = -1;
    w(3, 1) = w(4, 1) = -1;
    w(4, 2) = -1;

    REQUIRE(std::as_const(w) == wt);
}

TEST_CASE("cycle detection", "[CreateTransitiveDependencies]")
{
    xt::xtensor<int, 2> w = xt::zeros<int>({ 5, 5 });
    w(1, 0) = -1;
    w(2, 1) = -1;
    w(3, 2) = -1;
    w(4, 3) = -1;
    w(0, 4) = -1;

    REQUIRE_THROWS_AS(tsplp::CreateTransitiveDependencies(w), tsplp::CyclicDependenciesException);
}
