#include "GomoryHuTree.hpp"

#include "HasCycle.hpp"

#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/connected_components.hpp>
#include <catch2/catch.hpp>

#include <algorithm>
#include <array>
#include <span>
#include <unordered_map>
#include <utility>

using EdgeWeightProperty = boost::property<boost::edge_weight_t, double>;
using UndirectedGraph = boost::adjacency_list<
    boost::vecS, boost::vecS, boost::undirectedS, boost::no_property, EdgeWeightProperty>;
using VertexType = typename boost::graph_traits<UndirectedGraph>::vertex_descriptor;
using EdgeType = typename boost::graph_traits<UndirectedGraph>::edge_descriptor;

double GetMinCutFromGomoryHuTree(
    const UndirectedGraph& gomoryHuTree, VertexType source, VertexType sink)
{
    class Visitor : public boost::default_dfs_visitor
    {
        std::vector<VertexType>& m_predecessorMap;
        VertexType m_sink;

    public:
        Visitor(std::vector<VertexType>& predecessorMap, VertexType sink)
            : m_predecessorMap(predecessorMap)
            , m_sink(sink)
        {
        }

        void tree_edge(EdgeType e, const UndirectedGraph& g) const
        {
            const auto s = boost::source(e, g);
            const auto t = boost::target(e, g);
            m_predecessorMap[t] = s;
            if (t == m_sink)
                throw 0;
        }
    };

    std::vector<VertexType> predecessorMap(num_vertices(gomoryHuTree));
    try
    {
        static_assert(std::is_same_v<
                      typename boost::graph_traits<UndirectedGraph>::directed_category,
                      boost::undirected_tag>);
        std::unordered_map<EdgeType, boost::default_color_type, boost::hash<EdgeType>> colorMap;
        boost::undirected_dfs(
            gomoryHuTree,
            boost::root_vertex(source)
                .visitor(Visitor { predecessorMap, sink })
                .edge_color_map(boost::make_assoc_property_map(colorMap)));
    }
    catch (int)
    {
    }

    double minWeight = std::numeric_limits<double>::max();
    do
    {
        const auto [e, exists] = edge(predecessorMap[sink], sink, gomoryHuTree);
        if (!exists) // should not happen because predecessorMap was created from existing edges
            break;
        const auto weight = get(boost::edge_weight, gomoryHuTree, e);
        minWeight = std::min(minWeight, weight);
        sink = predecessorMap[sink];
    } while (sink != source);

    return minWeight;
}

namespace g = graph_algos;

template <size_t N>
class GomoryHuTreeCallback
{
    UndirectedGraph& m_gomoryHuTree;
    const std::array<std::array<int, N>, N>& m_expectedMinCuts;

public:
    GomoryHuTreeCallback(
        UndirectedGraph& gomoryHuTree, const std::array<std::array<int, N>, N>& expectedMinCuts)
        : m_gomoryHuTree(gomoryHuTree)
        , m_expectedMinCuts(expectedMinCuts)
    {
        REQUIRE(num_vertices(m_gomoryHuTree) == N);
    }

    bool operator()(
        VertexType u, VertexType v, double cutSize, std::span<const VertexType> compU,
        std::span<const VertexType> compV)
    {
        REQUIRE(N > 1); // the callback must not be called otherwise

        REQUIRE(u != v);
        REQUIRE(compU.size() + compV.size() == N);

        REQUIRE(m_expectedMinCuts.at(u).at(v) == cutSize);
        REQUIRE(m_expectedMinCuts.at(v).at(u) == cutSize);

        REQUIRE(std::find(compU.begin(), compU.end(), u) != compU.end());
        REQUIRE(std::find(compV.begin(), compV.end(), v) != compV.end());

        const auto [newEdge, wasAdded] = add_edge(u, v, cutSize, m_gomoryHuTree);
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
        REQUIRE(num_edges(m_gomoryHuTree) == (N == 0 ? 0 : N - 1));

        std::array<size_t, N> componentIds {};
        REQUIRE(
            boost::connected_components(m_gomoryHuTree, componentIds.data()) == (N == 0 ? 0 : 1));

        REQUIRE(!g::HasCycle(m_gomoryHuTree));
    }

    void CheckExpectedMinCuts() const
    {
        for (const auto u : boost::make_iterator_range(vertices(m_gomoryHuTree)))
        {
            for (const auto v : boost::make_iterator_range(vertices(m_gomoryHuTree)))
            {
                if (u == v)
                    continue;

                const auto minCut = GetMinCutFromGomoryHuTree(m_gomoryHuTree, u, v);
                REQUIRE(minCut == m_expectedMinCuts.at(u).at(v));
            }
        }
    }
};

auto GetLowerTriangularWeights(const UndirectedGraph& graph)
{
    const auto N = std::max(num_vertices(graph), static_cast<size_t>(1));
    std::vector<double> weights(N * (N - 1) / 2);
    for (const auto e : boost::make_iterator_range(edges(graph)))
    {
        const auto s = source(e, graph);
        const auto t = target(e, graph);
        const auto u = std::max(s, t);
        const auto v = std::min(s, t);

        weights[u * (u - 1) / 2 + v] = get(boost::edge_weight, graph, e);
    }

    return weights;
}

TEST_CASE("Empty Graph", "[Gomory Hu Tree]")
{
    constexpr int N = 0;
    UndirectedGraph graph(N);
    std::array<std::array<int, N>, N> expectedMinCuts {};

    UndirectedGraph gomoryHuTree(N);
    GomoryHuTreeCallback callback { gomoryHuTree, expectedMinCuts };

    g::CreateGomoryHuTree(N, GetLowerTriangularWeights(graph), callback);
    callback.DoPostCheck();
}

TEST_CASE("Single Node Graph", "[Gomory Hu Tree]")
{
    constexpr int N = 1;
    UndirectedGraph graph(N);
    constexpr std::array<std::array<int, N>, N> expectedMinCuts { { { 0 } } };

    UndirectedGraph gomoryHuTree(N);
    GomoryHuTreeCallback callback { gomoryHuTree, expectedMinCuts };

    g::CreateGomoryHuTree(N, GetLowerTriangularWeights(graph), callback);
    callback.DoPostCheck();
}

TEST_CASE("Two Nodes Graph", "[Gomory Hu Tree]")
{
    constexpr int N = 2;
    UndirectedGraph graph(N);

    add_edge(0, 1, 17, graph);

    // clang-format off
    constexpr std::array<std::array<int, N>, N> expectedMinCuts {{
        {  0, 17 },
        { 17,  0 },
    }};
    // clang-format on

    UndirectedGraph gomoryHuTree(N);
    GomoryHuTreeCallback callback { gomoryHuTree, expectedMinCuts };

    g::CreateGomoryHuTree(N, GetLowerTriangularWeights(graph), callback);
    callback.DoPostCheck();
}

TEST_CASE("Two Nodes Disjoint Graph", "[Gomory Hu Tree]")
{
    constexpr int N = 2;
    UndirectedGraph graph(N);

    // clang-format off
    constexpr std::array<std::array<int, N>, N> expectedMinCuts {{
        { 0, 0 },
        { 0, 0 },
    }};
    // clang-format on

    UndirectedGraph gomoryHuTree(N);
    GomoryHuTreeCallback callback { gomoryHuTree, expectedMinCuts };

    g::CreateGomoryHuTree(N, GetLowerTriangularWeights(graph), callback);
    callback.DoPostCheck();
}

TEST_CASE("Four Nodes Disjoint Graph", "[Gomory Hu Tree]")
{
    constexpr int N = 4;
    UndirectedGraph graph(N);

    // clang-format off
    constexpr std::array<std::array<int, N>, N> expectedMinCuts {{
        { 0, 0, 0, 0 },
        { 0, 0, 0, 0 },
        { 0, 0, 0, 0 },
        { 0, 0, 0, 0 },
    }};
    // clang-format on

    UndirectedGraph gomoryHuTree(N);
    GomoryHuTreeCallback callback { gomoryHuTree, expectedMinCuts };

    g::CreateGomoryHuTree(N, GetLowerTriangularWeights(graph), callback);
    callback.DoPostCheck();
}

TEST_CASE("K3", "[Gomory Hu Tree]")
{
    constexpr int N = 3;
    UndirectedGraph graph(N);

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

    UndirectedGraph gomoryHuTree(N);
    GomoryHuTreeCallback callback { gomoryHuTree, expectedMinCuts };

    g::CreateGomoryHuTree(N, GetLowerTriangularWeights(graph), callback);
    callback.DoPostCheck();
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

    // clang-format off
    constexpr std::array<std::array<int, N>, N> expectedMinCuts {{
        { 0, 7, 7, 7 },
        { 7, 0, 8,10 },
        { 7, 8, 0, 8 },
        { 7,10, 8, 0 },
    }};
    // clang-format on

    UndirectedGraph gomoryHuTree(N);
    GomoryHuTreeCallback callback { gomoryHuTree, expectedMinCuts };

    g::CreateGomoryHuTree(N, GetLowerTriangularWeights(graph), callback);
    callback.DoPostCheck();
}

TEST_CASE("Wikipedia example", "[Gomory Hu Tree]")
{
    // graph taken from
    // https://en.wikipedia.org/w/index.php?title=Gomory%E2%80%93Hu_tree&oldid=1097322015

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

    UndirectedGraph gomoryHuTree(N);
    GomoryHuTreeCallback callback { gomoryHuTree, expectedMinCuts };

    g::CreateGomoryHuTree(N, GetLowerTriangularWeights(graph), callback);
    callback.DoPostCheck();
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

    UndirectedGraph gomoryHuTree(N);
    GomoryHuTreeCallback callback { gomoryHuTree, expectedMinCuts };

    g::CreateGomoryHuTree(N, GetLowerTriangularWeights(graph), callback);
    callback.DoPostCheck();
}

TEST_CASE("Two connected components Graph", "[Gomory Hu Tree]")
{
    constexpr int N = 4;
    UndirectedGraph graph(N);

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

    UndirectedGraph gomoryHuTree(N);
    GomoryHuTreeCallback callback { gomoryHuTree, expectedMinCuts };

    g::CreateGomoryHuTree(N, GetLowerTriangularWeights(graph), callback);
    callback.DoPostCheck();
}

TEST_CASE("Stoer-Wagner Regression Test", "[Gomory Hu Tree]")
{
    // Making sure we don't suffer from a bug in stoer_wagner_min_cut that existed for a long time.
    // Using the same sample graph here that was used as a regression test for the fix:
    // https://github.com/boostorg/graph/issues/286

    constexpr int N = 8;
    const std::array<std::pair<int, int>, 13> edges { { { 0, 1 },
                                                        { 0, 2 },
                                                        { 0, 3 },
                                                        { 1, 2 },
                                                        { 1, 3 },
                                                        { 2, 3 },
                                                        { 4, 5 },
                                                        { 4, 6 },
                                                        { 4, 7 },
                                                        { 5, 6 },
                                                        { 5, 7 },
                                                        { 6, 7 },
                                                        { 0, 4 } } };
    const std::array<int, 13> ws = { 3, 3, 3, 2, 2, 2, 3, 3, 3, 2, 2, 2, 6 };
    UndirectedGraph graph(edges.begin(), edges.end(), ws.begin(), N, edges.size());

    UndirectedGraph gomoryHuTree(N);
    g::CreateGomoryHuTree(
        N, GetLowerTriangularWeights(graph),
        [&](VertexType u, VertexType v, double cutSize, auto, auto)
        {
            add_edge(u, v, cutSize, gomoryHuTree);
            return false;
        });
    REQUIRE(num_vertices(gomoryHuTree) == N);
    REQUIRE(num_edges(gomoryHuTree) == N - 1);

    const auto minCutByTree = GetMinCutFromGomoryHuTree(gomoryHuTree, 0, 4);

    REQUIRE(minCutByTree == 6); // the bug caused this to be 7
}
