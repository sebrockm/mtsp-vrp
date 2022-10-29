#include "GomoryHuTree.hpp"

#include "PartiallyContractedGraph.hpp"

#include <boost/graph/boykov_kolmogorov_max_flow.hpp>
#include <boost/graph/connected_components.hpp>
#include <boost/graph/filtered_graph.hpp>
#include <boost/graph/undirected_dfs.hpp>
#include <boost/range/iterator_range.hpp>

#include <unordered_map>
#include <vector>

namespace graph_algos
{

using IntermediateTree = boost::adjacency_list<
    boost::vecS, boost::vecS, boost::undirectedS, boost::no_property, EdgeWeightProperty>;
using TreeVertexType = typename boost::graph_traits<IntermediateTree>::vertex_descriptor;
using TreeEdgeType = typename boost::graph_traits<IntermediateTree>::edge_descriptor;

using PartiallyContractedGraphEdgeType = PartiallyContractedGraph::edge_descriptor;
using PartiallyContractedGraphVertexType = PartiallyContractedGraph::vertex_descriptor;

void CreateGomoryHuTree(
    const UndirectedGraph& inputGraph,
    std::function<bool(
        VertexType u, VertexType v, double cutSize, std::span<const VertexType> compU,
        std::span<const VertexType> compV)>
        newEdgeCallback)
{
    const auto N = num_vertices(inputGraph);
    if (N <= 1)
        return;

    std::vector<PartiallyContractedGraphVertexType> inputVertex2partiallyContractedMap(N);
    std::vector<size_t> gomoryHuForestVertex2ComponentIdMap(N);

    std::vector<std::span<VertexType>> gomoryHuTreeContractedVertices(N);

    PartiallyContractedGraph partiallyContractedGraph(N);
    IntermediateTree gomoryHuTree;
    const auto firstTreeVertex = add_vertex(gomoryHuTree);

    const auto [graphVerticesBegin, graphVerticesEnd] = vertices(inputGraph);
    std::vector<VertexType> inputGraphVertexStorage(graphVerticesBegin, graphVerticesEnd);
    std::vector<VertexType> gomoryHuTreeContractedVerticesStorage = inputGraphVertexStorage;

    gomoryHuTreeContractedVertices[firstTreeVertex]
        = { gomoryHuTreeContractedVerticesStorage.data(),
            gomoryHuTreeContractedVerticesStorage.size() };

    std::vector<TreeVertexType> treeNodesToBeSplit;
    treeNodesToBeSplit.push_back(firstTreeVertex);

    while (!treeNodesToBeSplit.empty())
    {
        const auto splitNode = treeNodesToBeSplit.back();
        treeNodesToBeSplit.pop_back();

        struct Filter
        {
            TreeVertexType splitNode;
            bool operator()(TreeVertexType v) const { return splitNode != v; }
        };
        const auto gomoryHuForest
            = make_filtered_graph(gomoryHuTree, boost::keep_all {}, Filter { splitNode });
        const auto numberOfComponents = boost::connected_components(
            gomoryHuForest, gomoryHuForestVertex2ComponentIdMap.data());

        const auto n = numberOfComponents + gomoryHuTreeContractedVertices[splitNode].size();
        partiallyContractedGraph.Reset(n);

        // The nodes [0, numberOfComponents[ are the contracted nodes
        // Fill in the subnodes into each contracted node
        for (const auto forestVertex : boost::make_iterator_range(vertices(gomoryHuForest)))
        {
            const auto componentId = gomoryHuForestVertex2ComponentIdMap[forestVertex];

            for (const auto v : gomoryHuTreeContractedVertices[forestVertex])
            {
                inputVertex2partiallyContractedMap[v] = componentId;
            }
        }

        // Copy internal nodes of split node into partially contracted graph.
        // These are the non contracted nodes in the range [numberOfComponents, ...[.
        [&]
        {
            PartiallyContractedGraph::vertex_descriptor nonContractedNode = numberOfComponents;
            for (const auto v : gomoryHuTreeContractedVertices[splitNode])
            {
                inputVertex2partiallyContractedMap[v] = nonContractedNode++;
            }
        }();

        // Fill in edges between contracted nodes by summing up original edges. We need forward and
        // backward edges because boost::boykov_kolmogorov_max_flow needs a directed input graph.
        for (const auto& inputEdge : boost::make_iterator_range(edges(inputGraph)))
        {
            const auto inputU = source(inputEdge, inputGraph);
            const auto inputV = target(inputEdge, inputGraph);

            const auto u = inputVertex2partiallyContractedMap[inputU];
            const auto v = inputVertex2partiallyContractedMap[inputV];

            if (u == v)
                continue;

            const auto weight = get(boost::edge_weight, inputGraph, inputEdge);
            partiallyContractedGraph.EdgeCapacities[u * n + v] += weight;
            partiallyContractedGraph.EdgeCapacities[v * n + u] += weight;
        }

        // finally, calculate the cut between some arbitrary non contracted nodes
        const auto inputSource = gomoryHuTreeContractedVertices[splitNode].front();
        const auto inputSink = gomoryHuTreeContractedVertices[splitNode].back();

        const auto source = inputVertex2partiallyContractedMap.at(inputSource);
        const auto sink = inputVertex2partiallyContractedMap.at(inputSink);

        const auto cutSize
            = boost::boykov_kolmogorov_max_flow(partiallyContractedGraph, source, sink);

        // add new vertex; splitNode will be split by moving ~ half of its nodes and edges here
        const auto newGomoryHuVertex = add_vertex(gomoryHuTree);

        // have the black nodes first
        const auto middle = std::partition(
            gomoryHuTreeContractedVertices[splitNode].begin(),
            gomoryHuTreeContractedVertices[splitNode].end(),
            [&](VertexType v)
            {
                const auto nonContractedNode = inputVertex2partiallyContractedMap.at(v);
                return partiallyContractedGraph.VertexColors[nonContractedNode]
                    != boost::white_color;
            });
        const auto splitOffset
            = static_cast<size_t>(middle - gomoryHuTreeContractedVertices[splitNode].begin());
        const auto blackNodes = gomoryHuTreeContractedVertices[splitNode].subspan(0, splitOffset);
        const auto whiteNodes = gomoryHuTreeContractedVertices[splitNode].subspan(splitOffset);

        // Split the split node: Assign all black subnodes to the new node
        gomoryHuTreeContractedVertices[newGomoryHuVertex] = blackNodes;

        // Distribute the edges of the split node: Copy all edges adjacent to a black node to the
        // new node
        std::vector<TreeEdgeType> edgesToRemove;
        for (const auto e : boost::make_iterator_range(out_edges(splitNode, gomoryHuTree)))
        {
            const auto targetNode = target(e, gomoryHuTree);
            const auto sampleInputVertex = gomoryHuTreeContractedVertices[targetNode].front();
            const auto sampleVertex = inputVertex2partiallyContractedMap.at(sampleInputVertex);

            if (partiallyContractedGraph.VertexColors[sampleVertex] != boost::white_color)
            {
                const auto weight = get(boost::edge_weight, gomoryHuTree, e);
                const auto [copiedEdge, _] = add_edge(newGomoryHuVertex, targetNode, gomoryHuTree);
                put(boost::edge_weight, gomoryHuTree, copiedEdge, weight);
                edgesToRemove.push_back(e);
            }
        }

        // Remove those edges that were previously copied over to the new node
        // It's not safe to remove them in the previous loop (while iterating them)
        for (const auto e : edgesToRemove)
            remove_edge(e, gomoryHuTree);

        // Remove the black nodes from the split node that were just copied over to the new node
        gomoryHuTreeContractedVertices[splitNode] = whiteNodes;

        // most important step: connect the now fully split nodes with the cut size
        add_edge(newGomoryHuVertex, splitNode, cutSize, gomoryHuTree);

        const auto endBlack = std::partition(
            begin(inputGraphVertexStorage), end(inputGraphVertexStorage),
            [&](VertexType v)
            {
                const auto pv = inputVertex2partiallyContractedMap.at(v);
                return partiallyContractedGraph.VertexColors[pv] != boost::white_color;
            });
        const auto blackLength = static_cast<size_t>(endBlack - begin(inputGraphVertexStorage));

        const auto isStopRequested = newEdgeCallback(
            inputSource, inputSink, cutSize,
            std::span { inputGraphVertexStorage.data(), blackLength },
            std::span { inputGraphVertexStorage.data() + blackLength,
                        inputGraphVertexStorage.size() - blackLength });
        if (isStopRequested)
            return;

        if (gomoryHuTreeContractedVertices[splitNode].size() > 1)
            treeNodesToBeSplit.push_back(splitNode);
        if (gomoryHuTreeContractedVertices[newGomoryHuVertex].size() > 1)
            treeNodesToBeSplit.push_back(newGomoryHuVertex);
    }
}

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

}
