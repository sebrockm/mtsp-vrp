#include "GomoryHuTree.hpp"

#include <boost/graph/adjacency_matrix.hpp>
#include <boost/graph/boykov_kolmogorov_max_flow.hpp>
#include <boost/graph/connected_components.hpp>
#include <boost/graph/depth_first_search.hpp>
#include <boost/graph/filtered_graph.hpp>
#include <boost/range/iterator_range.hpp>

#include <span>
#include <vector>

namespace tsplp::graph
{

using IntermediateTree = boost::adjacency_list<
    boost::vecS, boost::vecS, boost::undirectedS, boost::no_property, EdgeWeightProperty>;
using TreeVertexType = typename boost::graph_traits<IntermediateTree>::vertex_descriptor;
using TreeEdgeType = typename boost::graph_traits<IntermediateTree>::edge_descriptor;

using PartiallyContractedGraphEdgeType =
    typename boost::adjacency_matrix_traits<boost::directedS>::edge_descriptor;
using PartiallyContractedGraphVertexType =
    typename boost::adjacency_matrix_traits<boost::directedS>::vertex_descriptor;

using PartiallyContractedGraph = boost::adjacency_matrix<boost::directedS>;

void CreateGomoryHuTree(
    const UndirectedGraph& inputGraph,
    std::function<
        bool(double cutSize, std::vector<VertexType> comp1, std::vector<VertexType> comp2)>
        newEdgeCallback)
{
    const auto N = num_vertices(inputGraph);
    if (N <= 1)
        return;

    std::vector<PartiallyContractedGraphVertexType> inputVertex2partiallyContractedMap(N);
    std::vector<size_t> gomoryHuForestVertex2ComponentIdMap(N);

    std::vector<boost::default_color_type> partiallyContractedGraphColor(N);
    std::vector<PartiallyContractedGraphEdgeType> partiallyContractedGraphPredecessorEdge(N);
    std::vector<double> partiallyContractedGraphDistance(N);

    std::vector<double> partiallyContractedGraphCapacity(N * N);
    std::vector<double> partiallyContractedGraphResidualCapacity(N * N);
    std::vector<PartiallyContractedGraphEdgeType> partiallyContractedGraphReverseEdge(N * N);

    const auto GetEdgePropertyFunctionMap = [](auto& matrix, auto& graph)
    {
        return boost::make_function_property_map<PartiallyContractedGraphEdgeType>(
            [&](const PartiallyContractedGraphEdgeType& e) -> auto&
            {
                const auto n = num_vertices(graph);
                const auto s = source(e, graph);
                const auto t = target(e, graph);
                return matrix[n * s + t];
            });
    };

    std::vector<std::span<VertexType>> gomoryHuTreeContractedVertices(N);

    IntermediateTree gomoryHuTree;
    const auto firstTreeVertex = add_vertex(gomoryHuTree);

    const auto [graphVerticesBegin, graphVerticesEnd] = vertices(inputGraph);
    std::vector<VertexType> gomoryHuTreeContractedVerticesStorage(
        graphVerticesBegin, graphVerticesEnd);
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
        PartiallyContractedGraph partiallyContractedGraph(n);
        for (size_t i = 0; i < n * n; ++i)
            partiallyContractedGraphCapacity[i] = 0;

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
        PartiallyContractedGraphVertexType nonContractedNode = numberOfComponents;
        for (const auto v : gomoryHuTreeContractedVertices[splitNode])
        {
            inputVertex2partiallyContractedMap[v] = nonContractedNode++;
        }

        // Fill in edges between contracted nodes by summing up original edges. We need forward and
        // backward edges because boost::boykov_kolmogorov_max_flow needs a directed input graph.
        for (const auto& inputEdge : boost::make_iterator_range(edges(inputGraph)))
        {
            const auto inputU = source(inputEdge, inputGraph);
            const auto inputV = target(inputEdge, inputGraph);

            const auto u = inputVertex2partiallyContractedMap[inputU];
            const auto v = inputVertex2partiallyContractedMap[inputV];

            const auto [forwardEdge, forwardInserted] = add_edge(u, v, partiallyContractedGraph);
            const auto [backwardEdge, backwardInserted] = add_edge(v, u, partiallyContractedGraph);
            assert(forwardInserted);
            assert(backwardInserted);

            const auto weight = get(boost::edge_weight, inputGraph, inputEdge);
            partiallyContractedGraphCapacity[u * n + v] += weight;
            partiallyContractedGraphCapacity[v * n + u] += weight;
            partiallyContractedGraphReverseEdge[u * n + v] = backwardEdge;
            partiallyContractedGraphReverseEdge[v * n + u] = forwardEdge;
        }

        // finally, calculate the cut between some arbitrary non contracted nodes
        const auto inputSource = gomoryHuTreeContractedVertices[splitNode].front();
        const auto inputSink = gomoryHuTreeContractedVertices[splitNode].back();

        const auto source = inputVertex2partiallyContractedMap.at(inputSource);
        const auto sink = inputVertex2partiallyContractedMap.at(inputSink);

        const auto vertexIndexMap = get(boost::vertex_index, partiallyContractedGraph);

        const auto cutSize = boost::boykov_kolmogorov_max_flow(
            partiallyContractedGraph,
            GetEdgePropertyFunctionMap(partiallyContractedGraphCapacity, partiallyContractedGraph),
            GetEdgePropertyFunctionMap(
                partiallyContractedGraphResidualCapacity, partiallyContractedGraph),
            GetEdgePropertyFunctionMap(
                partiallyContractedGraphReverseEdge, partiallyContractedGraph),
            boost::make_iterator_property_map(
                partiallyContractedGraphPredecessorEdge.data(), vertexIndexMap),
            boost::make_iterator_property_map(partiallyContractedGraphColor.data(), vertexIndexMap),
            boost::make_iterator_property_map(
                partiallyContractedGraphDistance.data(), vertexIndexMap),
            vertexIndexMap, source, sink);

        // add new vertex; splitNode will be split by moving ~ half of its nodes and edges here
        const auto newGomoryHuVertex = add_vertex(gomoryHuTree);

        // have the black nodes first
        const auto middle = std::partition(
            gomoryHuTreeContractedVertices[splitNode].begin(),
            gomoryHuTreeContractedVertices[splitNode].end(),
            [&](VertexType v)
            {
                const auto nonContractedNode = inputVertex2partiallyContractedMap.at(v);
                return partiallyContractedGraphColor[nonContractedNode] == boost::black_color;
            });
        const auto splitOffset = static_cast<size_t>(middle - gomoryHuTreeContractedVertices[splitNode].begin());
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

            if (partiallyContractedGraphColor[sampleVertex] == boost::black_color)
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

        std::vector<VertexType> comp1, comp2;
        for (const auto v : boost::make_iterator_range(vertices(inputGraph)))
        {
            const auto pv = inputVertex2partiallyContractedMap.at(v);
            if (partiallyContractedGraphColor[pv] == boost::black_color)
                comp1.push_back(v);
            else
                comp2.push_back(v);
        }
        const auto isStopRequested = newEdgeCallback(cutSize, std::move(comp1), std::move(comp2));
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
        boost::depth_first_search(
            gomoryHuTree, boost::root_vertex(source).visitor(Visitor { predecessorMap, sink }));
    }
    catch (int)
    {
    }

    double minWeight = std::numeric_limits<double>::max();
    do
    {
        const auto [e, _] = edge(predecessorMap[sink], sink, gomoryHuTree);
        const auto weight = get(boost::edge_weight, gomoryHuTree, e);
        minWeight = std::min(minWeight, weight);
        sink = predecessorMap[sink];
    } while (sink != source);

    return minWeight;
}

}
