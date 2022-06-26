#include "GomoryHuTree.hpp"

#include <catch2/catch.hpp>

TEST_CASE("K3", "[Gomory Hu Tree]")
{
    constexpr int N = 3;
    UndirectedGraph graph(N);

    add_edge(0, 1, 1, graph);
    add_edge(0, 2, 2, graph);
    add_edge(1, 2, 4, graph);

    const auto gomoryHuTree = CreateGomoryHuTree(graph);

    REQUIRE(num_vertices(gomoryHuTree) == N);
    REQUIRE(num_edges(gomoryHuTree) == N - 1);

    // clang-format off
    constexpr std::array<std::array<int, N>, N> expectedMinCuts {{
        { 0, 3, 3 },
        { 3, 0, 5 },
        { 3, 5, 0 },
    }};
    // clang-format on

    for (const auto u : boost::make_iterator_range(vertices(graph)))
    {
        for (const auto v : boost::make_iterator_range(vertices(graph)))
        {
            if (u == v)
                continue;

            const auto minCutByTree = GetMinCutFromGomoryHuTree(gomoryHuTree, u, v);

            REQUIRE(minCutByTree == expectedMinCuts[u][v]);
        }
    }
}

TEST_CASE("K4", "[Gomory Hu Tree]")
{
    constexpr int N = 4;
    UndirectedGraph graph(N);

    add_edge(0, 1, 1, graph);
    add_edge(0, 2, 2, graph);
    add_edge(0, 3, 4, graph);
    add_edge(1, 2, 4, graph);
    add_edge(1, 3, 5, graph);
    add_edge(2, 3, 2, graph);

    const auto gomoryHuTree = CreateGomoryHuTree(graph);

    REQUIRE(num_vertices(gomoryHuTree) == N);
    REQUIRE(num_edges(gomoryHuTree) == N - 1);

    // clang-format off
    constexpr std::array<std::array<int, N>, N> expectedMinCuts {{
        { 0, 7, 7, 7 },
        { 7, 0, 8,10 },
        { 7, 8, 0, 8 },
        { 7,10, 8, 0 },
    }};
    // clang-format on

    for (const auto u : boost::make_iterator_range(vertices(graph)))
    {
        for (const auto v : boost::make_iterator_range(vertices(graph)))
        {
            if (u == v)
                continue;

            const auto minCutByTree = GetMinCutFromGomoryHuTree(gomoryHuTree, u, v);

            REQUIRE(minCutByTree == expectedMinCuts[u][v]);
        }
    }
}

TEST_CASE("Wikipedia example", "[Gomory Hu Tree]")
{
    constexpr int N = 6;
    UndirectedGraph graph(N);

    add_edge(0, 1, 1, graph);
    add_edge(0, 2, 7, graph);
    add_edge(1, 3, 3, graph);
    add_edge(1, 4, 2, graph);
    add_edge(1, 2, 1, graph);
    add_edge(2, 4, 4, graph);
    add_edge(3, 4, 1, graph);
    add_edge(3, 5, 6, graph);
    add_edge(4, 5, 2, graph);

    const auto gomoryHuTree = CreateGomoryHuTree(graph);

    REQUIRE(num_vertices(gomoryHuTree) == N);
    REQUIRE(num_edges(gomoryHuTree) == N - 1);

    // clang-format off
    constexpr std::array<std::array<int, N>, N> expectedMinCuts {{
        { 0, 6, 8, 6, 6, 6 },
        { 6, 0, 6, 6, 7, 6 },
        { 8, 6, 0, 6, 6, 6 },
        { 6, 6, 6, 0, 6, 8 },
        { 6, 7, 6, 6, 0, 6 },
        { 6, 6, 6, 8, 6, 0 },
    }};
    // clang-format on

    for (const auto u : boost::make_iterator_range(vertices(graph)))
    {
        for (const auto v : boost::make_iterator_range(vertices(graph)))
        {
            if (u == v)
                continue;

            const auto minCutByTree = GetMinCutFromGomoryHuTree(gomoryHuTree, u, v);

            REQUIRE(minCutByTree == expectedMinCuts[u][v]);
        }
    }
}

TEST_CASE("Lecture example", "[Gomory Hu Tree]")
{
    // graph taken from http://www14.in.tum.de/lehre/2016WS/ea/split/sec-Gomory-Hu-Trees.pdf

    constexpr int N = 9;
    UndirectedGraph graph(N);

    add_edge(0, 1, 2, graph);
    add_edge(0, 2, 4, graph);
    add_edge(0, 6, 1, graph);

    add_edge(1, 2, 6, graph);
    add_edge(1, 3, 11, graph);

    add_edge(2, 4, 9, graph);

    add_edge(3, 4, 7, graph);
    add_edge(3, 6, 2, graph);

    add_edge(4, 5, 9, graph);
    add_edge(4, 6, 3, graph);
    add_edge(4, 7, 1, graph);

    add_edge(5, 7, 8, graph);

    add_edge(6, 7, 9, graph);
    add_edge(6, 8, 4, graph);

    add_edge(7, 8, 3, graph);

    const auto gomoryHuTree = CreateGomoryHuTree(graph);

    REQUIRE(num_vertices(gomoryHuTree) == N);
    REQUIRE(num_edges(gomoryHuTree) == N - 1);

    // clang-format off
    constexpr std::array<std::array<int, N>, N> expectedMinCuts {{
        { 0,  7,  7,  7,  7,  7,  7,  7, 7 },
        { 7,  0, 17, 19, 17, 16, 15, 15, 7 },
        { 7, 17,  0, 17, 18, 16, 15, 15, 7 },
        { 7, 19, 17,  0, 17, 16, 15, 15, 7 },
        { 7, 17, 18, 17,  0, 16, 15, 15, 7 },
        { 7, 16, 16, 16, 16,  0, 15, 15, 7 },
        { 7, 15, 15, 15, 15, 15,  0, 18, 7 },
        { 7, 15, 15, 15, 15, 15, 18,  0, 7 },
        { 7,  7,  7,  7,  7,  7,  7,  7, 0 },
    }};
    // clang-format on

    for (const auto u : boost::make_iterator_range(vertices(graph)))
    {
        for (const auto v : boost::make_iterator_range(vertices(graph)))
        {
            if (u == v)
                continue;

            const auto minCutByTree = GetMinCutFromGomoryHuTree(gomoryHuTree, u, v);

            REQUIRE(minCutByTree == expectedMinCuts[u][v]);
        }
    }
}
