#include "GomoryHuTree.hpp"

#include <boost/graph/connected_components.hpp>
#include <boost/graph/depth_first_search.hpp>
#include <catch2/catch.hpp>

namespace g = graph_algos;

template <size_t N>
class GomoryHuTreeCallback
{
    g::UndirectedGraph m_tree;
    const std::array<std::array<int, N>, N>& m_expectedMinCuts;

public:
    GomoryHuTreeCallback(const std::array<std::array<int, N>, N>& expectedMinCuts)
        : m_tree(N)
        , m_expectedMinCuts(expectedMinCuts)
    {
    }

    bool operator()(
        g::VertexType u, g::VertexType v, double cutSize, std::span<const g::VertexType> compU,
        std::span<const g::VertexType> compV)
    {
        REQUIRE(N > 1); // the callback must not be called otherwise

        REQUIRE(u != v);
        REQUIRE(compU.size() + compV.size() == N);

        REQUIRE(m_expectedMinCuts[u][v] == cutSize);
        REQUIRE(m_expectedMinCuts[v][u] == cutSize);

        REQUIRE(std::find(compU.begin(), compU.end(), u) != compU.end());
        REQUIRE(std::find(compV.begin(), compV.end(), v) != compV.end());

        const auto [newEdge, wasAdded] = add_edge(u, v, cutSize, m_tree);
        REQUIRE(wasAdded);

        return false; // don't interrupt
    }

    void DoPostCheck() const
    {
        CheckIfTree();
        CheckExpectedMinCuts();
    }

private:
    void CheckIfTree() const
    {
        REQUIRE(num_edges(m_tree) == (N == 0 ? 0 : N - 1));

        std::array<size_t, N> componentIds;
        REQUIRE(boost::connected_components(m_tree, componentIds.data()) == (N == 0 ? 0 : 1));

        const auto hasCircle = [&]
        {
            struct CircleFinder : boost::default_dfs_visitor
            {
                void back_edge(g::EdgeType, const g::UndirectedGraph&) const { throw 0; }
            };
            try
            {
                boost::depth_first_search(m_tree, boost::visitor(CircleFinder {}));
            }
            catch (int)
            {
                return true;
            }
            return false;
        }();
        REQUIRE(!hasCircle);
    }

    void CheckExpectedMinCuts() const
    {
        for (const auto u : boost::make_iterator_range(vertices(m_tree)))
        {
            for (const auto v : boost::make_iterator_range(vertices(m_tree)))
            {
                if (u == v)
                    continue;

                const auto minCut = g::GetMinCutFromGomoryHuTree(m_tree, u, v);
                REQUIRE(minCut == m_expectedMinCuts[u][v]);
            }
        }
    }
};

TEST_CASE("Empty Graph", "[Gomory Hu Tree]")
{
    constexpr int N = 0;
    g::UndirectedGraph graph(N);
    std::array<std::array<int, N>, N> expectedMinCuts {};
    GomoryHuTreeCallback callback { expectedMinCuts };

    g::CreateGomoryHuTree(graph, callback);

    callback.DoPostCheck();
}

TEST_CASE("Single Node Graph", "[Gomory Hu Tree]")
{
    constexpr int N = 1;
    g::UndirectedGraph graph(N);
    constexpr std::array<std::array<int, N>, N> expectedMinCuts { { { 0 } } };
    GomoryHuTreeCallback callback { expectedMinCuts };

    g::CreateGomoryHuTree(graph, callback);

    callback.DoPostCheck();
}

TEST_CASE("Two Nodes Graph", "[Gomory Hu Tree]")
{
    constexpr int N = 2;
    g::UndirectedGraph graph(N);

    add_edge(0, 1, 17, graph);

    // clang-format off
    constexpr std::array<std::array<int, N>, N> expectedMinCuts {{
        {  0, 17 },
        { 17,  0 },
    }};
    // clang-format on

    GomoryHuTreeCallback callback { expectedMinCuts };

    g::CreateGomoryHuTree(graph, callback);

    callback.DoPostCheck();
}

TEST_CASE("Two Nodes Disjoint Graph", "[Gomory Hu Tree]")
{
    constexpr int N = 2;
    g::UndirectedGraph graph(N);

    // clang-format off
    constexpr std::array<std::array<int, N>, N> expectedMinCuts {{
        { 0, 0 },
        { 0, 0 },
    }};
    // clang-format on

    GomoryHuTreeCallback callback { expectedMinCuts };

    g::CreateGomoryHuTree(graph, callback);

    callback.DoPostCheck();
}

TEST_CASE("Four Nodes Disjoint Graph", "[Gomory Hu Tree]")
{
    constexpr int N = 4;
    g::UndirectedGraph graph(N);

    // clang-format off
    constexpr std::array<std::array<int, N>, N> expectedMinCuts {{
        { 0, 0, 0, 0 },
        { 0, 0, 0, 0 },
        { 0, 0, 0, 0 },
        { 0, 0, 0, 0 },
    }};
    // clang-format on

    GomoryHuTreeCallback callback { expectedMinCuts };

    g::CreateGomoryHuTree(graph, callback);

    callback.DoPostCheck();
}

TEST_CASE("K3", "[Gomory Hu Tree]")
{
    constexpr int N = 3;
    g::UndirectedGraph graph(N);

    add_edge(0, 1, 1, graph);
    add_edge(0, 2, 2, graph);
    add_edge(1, 2, 4, graph);

    // clang-format off
    constexpr std::array<std::array<int, N>, N> expectedMinCuts {{
        { 0, 3, 3 },
        { 3, 0, 5 },
        { 3, 5, 0 },
    }};
    // clang-format on

    GomoryHuTreeCallback callback { expectedMinCuts };

    g::CreateGomoryHuTree(graph, callback);

    callback.DoPostCheck();
}

TEST_CASE("K4", "[Gomory Hu Tree]")
{
    constexpr int N = 4;
    g::UndirectedGraph graph(N);

    add_edge(0, 1, 1, graph);
    add_edge(0, 2, 2, graph);
    add_edge(0, 3, 4, graph);
    add_edge(1, 2, 4, graph);
    add_edge(1, 3, 5, graph);
    add_edge(2, 3, 2, graph);

    // clang-format off
    constexpr std::array<std::array<int, N>, N> expectedMinCuts {{
        { 0, 7, 7, 7 },
        { 7, 0, 8,10 },
        { 7, 8, 0, 8 },
        { 7,10, 8, 0 },
    }};
    // clang-format on

    GomoryHuTreeCallback callback { expectedMinCuts };

    g::CreateGomoryHuTree(graph, callback);

    callback.DoPostCheck();
}

TEST_CASE("Wikipedia example", "[Gomory Hu Tree]")
{
    // graph taken from
    // https://en.wikipedia.org/w/index.php?title=Gomory%E2%80%93Hu_tree&oldid=1097322015

    constexpr int N = 6;
    g::UndirectedGraph graph(N);

    add_edge(0, 1, 1, graph);
    add_edge(0, 2, 7, graph);
    add_edge(1, 3, 3, graph);
    add_edge(1, 4, 2, graph);
    add_edge(1, 2, 1, graph);
    add_edge(2, 4, 4, graph);
    add_edge(3, 4, 1, graph);
    add_edge(3, 5, 6, graph);
    add_edge(4, 5, 2, graph);

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

    GomoryHuTreeCallback callback { expectedMinCuts };

    g::CreateGomoryHuTree(graph, callback);

    callback.DoPostCheck();
}

TEST_CASE("Lecture example", "[Gomory Hu Tree]")
{
    // graph taken from http://www14.in.tum.de/lehre/2016WS/ea/split/sec-Gomory-Hu-Trees.pdf

    constexpr int N = 9;
    g::UndirectedGraph graph(N);

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

    GomoryHuTreeCallback callback { expectedMinCuts };

    g::CreateGomoryHuTree(graph, callback);

    callback.DoPostCheck();
}

TEST_CASE("Two connected components Graph", "[Gomory Hu Tree]")
{
    constexpr int N = 4;
    g::UndirectedGraph graph(N);

    add_edge(0, 1, 1, graph);
    add_edge(2, 3, 1, graph);

    // clang-format off
    constexpr std::array<std::array<int, N>, N> expectedMinCuts {{
        { 0, 1, 0, 0 },
        { 1, 0, 0, 0 },
        { 0, 0, 0, 1 },
        { 0, 0, 1, 0 },
    }};
    // clang-format on

    GomoryHuTreeCallback callback { expectedMinCuts };

    g::CreateGomoryHuTree(graph, callback);

    callback.DoPostCheck();
}

TEST_CASE("Stoer-Wagner Regression Test", "[Gomory Hu Tree]")
{
    // Making sure we don't suffer from a bug in stoer_wagner_min_cut that existed for a long time.
    // Using the same sample graph here that was used as a regression test for the fix:
    // https://github.com/boostorg/graph/issues/286

    constexpr int N = 8;
    const std::pair<int, int> edges[]
        = { { 0, 1 }, { 0, 2 }, { 0, 3 }, { 1, 2 }, { 1, 3 }, { 2, 3 }, { 4, 5 },
            { 4, 6 }, { 4, 7 }, { 5, 6 }, { 5, 7 }, { 6, 7 }, { 0, 4 } };
    const int ws[] = { 3, 3, 3, 2, 2, 2, 3, 3, 3, 2, 2, 2, 6 };
    g::UndirectedGraph graph(edges, edges + 13, ws, N, 13);

    g::UndirectedGraph gomoryHuTree(N);
    g::CreateGomoryHuTree(
        graph,
        [&](g::VertexType u, g::VertexType v, double cutSize, auto, auto)
        {
            add_edge(u, v, cutSize, gomoryHuTree);
            return false;
        });
    REQUIRE(num_vertices(gomoryHuTree) == N);
    REQUIRE(num_edges(gomoryHuTree) == N - 1);

    const auto minCutByTree = g::GetMinCutFromGomoryHuTree(gomoryHuTree, 0, 4);

    REQUIRE(minCutByTree == 6); // the bug caused this to be 7
}
