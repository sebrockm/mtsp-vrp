#include "DependencyHelpers.hpp"

#include <catch2/catch.hpp>

TEST_CASE("empty", "[CreateTransitiveDependencies]") 
{
    const auto g = tsplp::graph::CreateTransitiveDependencies({ });

    REQUIRE(g.size() == 0);

    REQUIRE_THROWS(g.at(0));
}

TEST_CASE("single", "[CreateTransitiveDependencies]")
{
    const auto g = tsplp::graph::CreateTransitiveDependencies(std::vector{ std::pair{0, 1} });

    REQUIRE(g.size() == 1);
    REQUIRE(g.at(0).size() == 1);
    REQUIRE(g.at(0)[0] == 1);

    REQUIRE_THROWS(g.at(1));
}

TEST_CASE("single not first", "[CreateTransitiveDependencies]")
{
    const auto g = tsplp::graph::CreateTransitiveDependencies(std::vector{ std::pair{17, 100} });

    REQUIRE(g.size() == 1);
    REQUIRE(g.at(17).size() == 1);
    REQUIRE(g.at(17)[0] == 100);

    REQUIRE_THROWS(g.at(100));
}

TEST_CASE("transitive line", "[CreateTransitiveDependencies]")
{
    const auto g = tsplp::graph::CreateTransitiveDependencies(std::vector{ std::pair{0, 1}, std::pair{1, 2}, std::pair{2, 3}, std::pair{3, 4} });

    REQUIRE(g.size() == 4);

    REQUIRE(g.at(0).size() == 4);
    REQUIRE(g.at(0)[0] == 1);
    REQUIRE(g.at(0)[1] == 2);
    REQUIRE(g.at(0)[2] == 3);
    REQUIRE(g.at(0)[3] == 4);

    REQUIRE(g.at(1).size() == 3);
    REQUIRE(g.at(1)[0] == 2);
    REQUIRE(g.at(1)[1] == 3);
    REQUIRE(g.at(1)[2] == 4);

    REQUIRE(g.at(2).size() == 2);
    REQUIRE(g.at(2)[0] == 3);
    REQUIRE(g.at(2)[1] == 4);

    REQUIRE(g.at(3).size() == 1);
    REQUIRE(g.at(3)[0] == 4);

    REQUIRE_THROWS(g.at(-1));
    REQUIRE_THROWS(g.at(4));
    REQUIRE_THROWS(g.at(5));
    REQUIRE_THROWS(g.at(1234));
}

TEST_CASE("empty reversed", "[CreateTransitiveDependenciesReversed]")
{
    const auto g = tsplp::graph::CreateTransitiveDependenciesReversed({ });

    REQUIRE(g.size() == 0);

    REQUIRE_THROWS(g.at(0));
}

TEST_CASE("single reversed", "[CreateTransitiveDependenciesReversed]")
{
    const auto g = tsplp::graph::CreateTransitiveDependenciesReversed(std::vector{ std::pair{0, 1} });

    REQUIRE(g.size() == 1);
    REQUIRE(g.at(1).size() == 1);
    REQUIRE(g.at(1)[0] == 0);

    REQUIRE_THROWS(g.at(0));
}

TEST_CASE("single not first reversed", "[CreateTransitiveDependenciesReversed]")
{
    const auto g = tsplp::graph::CreateTransitiveDependenciesReversed(std::vector{ std::pair{17, 100} });

    REQUIRE(g.size() == 1);
    REQUIRE(g.at(100).size() == 1);
    REQUIRE(g.at(100)[0] == 17);

    REQUIRE_THROWS(g.at(17));
}

TEST_CASE("transitive line reversed", "[CreateTransitiveDependenciesReversed]")
{
    const auto g = tsplp::graph::CreateTransitiveDependenciesReversed(std::vector{ std::pair{0, 1}, std::pair{1, 2}, std::pair{2, 3}, std::pair{3, 4} });

    REQUIRE(g.size() == 4);

    REQUIRE(g.at(4).size() == 4);
    REQUIRE(g.at(4)[0] == 0);
    REQUIRE(g.at(4)[1] == 1);
    REQUIRE(g.at(4)[2] == 2);
    REQUIRE(g.at(4)[3] == 3);

    REQUIRE(g.at(3).size() == 3);
    REQUIRE(g.at(3)[0] == 0);
    REQUIRE(g.at(3)[1] == 1);
    REQUIRE(g.at(3)[2] == 2);

    REQUIRE(g.at(2).size() == 2);
    REQUIRE(g.at(2)[0] == 0);
    REQUIRE(g.at(2)[1] == 1);

    REQUIRE(g.at(1).size() == 1);
    REQUIRE(g.at(1)[0] == 0);

    REQUIRE_THROWS(g.at(-1));
    REQUIRE_THROWS(g.at(0));
    REQUIRE_THROWS(g.at(5));
    REQUIRE_THROWS(g.at(1234));
}