#include "GomoryHuTree.hpp"

#include <boost/graph/boykov_kolmogorov_max_flow.hpp>
#include <boost/graph/connected_components.hpp>
#include <boost/graph/depth_first_search.hpp>
#include <boost/graph/filtered_graph.hpp>
#include <boost/range/iterator_range.hpp>

#include <unordered_set>
#include <vector>

struct TreeVertexProperty
{
    std::unordered_set<VertexType> ContractedVertices;
};

using IntermediateTree = boost::adjacency_list<
    boost::vecS, boost::vecS, boost::undirectedS, TreeVertexProperty, EdgeWeightProperty>;
using TreeVertexType = typename boost::graph_traits<IntermediateTree>::vertex_descriptor;
using TreeEdgeType = typename boost::graph_traits<IntermediateTree>::edge_descriptor;

using PartiallyContractedGraphEdgeType = typename boost::adjacency_list_traits<
    boost::vecS, boost::vecS, boost::directedS>::edge_descriptor;
using PartiallyContractedGraphVertexType = typename boost::adjacency_list_traits<
    boost::vecS, boost::vecS, boost::directedS>::vertex_descriptor;

struct PartiallyContractedGraphVertexProperty
{
    std::unordered_set<VertexType> ContractedVertices;
    PartiallyContractedGraphEdgeType PredecessorEdge;
    boost::default_color_type Color;
    double Distance;
};

struct ParitallyContractedGraphEdgeProperty
{
    double Capacity {};
    double ResidualCapacity {};
    PartiallyContractedGraphEdgeType ReverseEdge {};
};

using PartiallyContractedGraph = boost::adjacency_list<
    boost::vecS, boost::vecS, boost::directedS, PartiallyContractedGraphVertexProperty,
    ParitallyContractedGraphEdgeProperty>;

UndirectedGraph CreateGomoryHuTree(const UndirectedGraph& inputGraph)
{
    const auto N = num_vertices(inputGraph);
    if (N <= 2)
        return inputGraph;

    IntermediateTree gomoryHuTree;
    const auto firstTreeVertex = add_vertex(gomoryHuTree);

    const auto [graphVerticesBegin, graphVerticesEnd] = vertices(inputGraph);
    gomoryHuTree[firstTreeVertex].ContractedVertices.insert(graphVerticesBegin, graphVerticesEnd);

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

        using ForestVertexType =
            typename boost::graph_traits<decltype(gomoryHuForest)>::vertex_descriptor;

        std::unordered_map<ForestVertexType, size_t> gomoryHuForestVertex2ComponentIdMap;
        const auto numberOfComponents = boost::connected_components(
            gomoryHuForest, boost::make_assoc_property_map(gomoryHuForestVertex2ComponentIdMap));

        PartiallyContractedGraph partiallyContractedGraph;

        // Copy internal nodes of split node into partially contracted graph.
        // These are the non contracted nodes.
        std::unordered_map<VertexType, PartiallyContractedGraphVertexType> // can be made a vector
            inputVertex2partiallyContractedMap;
        for (const auto v : gomoryHuTree[splitNode].ContractedVertices)
        {
            const auto nonContractedNode = add_vertex(partiallyContractedGraph);
            partiallyContractedGraph[nonContractedNode].ContractedVertices.insert(v);
            assert(!inputVertex2partiallyContractedMap.contains(v));
            inputVertex2partiallyContractedMap[v] = nonContractedNode;
        }

        // Add each connected component as a contracted node
        std::vector<PartiallyContractedGraphVertexType> contractedNodes;
        contractedNodes.reserve(numberOfComponents);
        for (size_t i = 0; i < numberOfComponents; ++i)
        {
            const auto contractedNode = add_vertex(partiallyContractedGraph);
            contractedNodes.push_back(contractedNode);
        }

        for (const auto [forestVertex, componentId] : gomoryHuForestVertex2ComponentIdMap)
        {
            const auto contractedNode = contractedNodes[componentId];
            const auto& contractedComponentNodes = gomoryHuForest[forestVertex].ContractedVertices;

            // fill in the subnodes into each contracted node
            for (const auto u : contractedComponentNodes)
            {
                partiallyContractedGraph[contractedNode].ContractedVertices.insert(u);
                assert(!inputVertex2partiallyContractedMap.contains(u));
                inputVertex2partiallyContractedMap[u] = contractedNode;
            }
        }

        // Fill in edges between contracted nodes by summing up original edges. We need forward and
        // backward edges because boost::boykov_kolmogorov_max_flow needs a directed input graph.
        for (const auto u : boost::make_iterator_range(vertices(partiallyContractedGraph)))
        {
            for (const auto v : boost::make_iterator_range(vertices(partiallyContractedGraph)))
            {
                if (v <= u)
                    continue;

                for (const auto inputU : partiallyContractedGraph[u].ContractedVertices)
                {
                    for (const auto inputV : partiallyContractedGraph[v].ContractedVertices)
                    {
                        if (const auto [inputEdge, exists] = edge(inputU, inputV, inputGraph);
                            exists)
                        {
                            const auto [forwardEdge, forwardInserted]
                                = add_edge(u, v, partiallyContractedGraph);
                            const auto [backwardEdge, backwardInserted]
                                = add_edge(v, u, partiallyContractedGraph);

                            const auto weight = get(boost::edge_weight, inputGraph, inputEdge);
                            partiallyContractedGraph[forwardEdge].Capacity += weight;
                            partiallyContractedGraph[backwardEdge].Capacity += weight;
                            partiallyContractedGraph[forwardEdge].ReverseEdge = backwardEdge;
                            partiallyContractedGraph[backwardEdge].ReverseEdge = forwardEdge;
                        }
                    }
                }
            }
        }

        // finally, calculate the cut between some arbitrary non contracted nodes
        const auto inputSource = *gomoryHuTree[splitNode].ContractedVertices.begin();
        const auto inputSink = *std::next(gomoryHuTree[splitNode].ContractedVertices.begin());

        const auto source = inputVertex2partiallyContractedMap.at(inputSource);
        const auto sink = inputVertex2partiallyContractedMap.at(inputSink);

        const auto cutSize = boost::boykov_kolmogorov_max_flow(
            partiallyContractedGraph,
            get(&ParitallyContractedGraphEdgeProperty::Capacity, partiallyContractedGraph),
            get(&ParitallyContractedGraphEdgeProperty::ResidualCapacity, partiallyContractedGraph),
            get(&ParitallyContractedGraphEdgeProperty::ReverseEdge, partiallyContractedGraph),
            get(&PartiallyContractedGraphVertexProperty::PredecessorEdge, partiallyContractedGraph),
            get(&PartiallyContractedGraphVertexProperty::Color, partiallyContractedGraph),
            get(&PartiallyContractedGraphVertexProperty::Distance, partiallyContractedGraph),
            get(boost::vertex_index, partiallyContractedGraph), source, sink);

        // add new vertex; splitNode will be split by moving ~ half of its nodes and edges here
        const auto newGomoryHuVertex = add_vertex(gomoryHuTree);

        // Split the split node: Copy all black subnodes to the new node
        for (const auto nodeInSplitNode : gomoryHuTree[splitNode].ContractedVertices)
        {
            const auto nonContractedNode = inputVertex2partiallyContractedMap.at(nodeInSplitNode);

            // copy black nodes over to the new gomory hu vertex
            if (partiallyContractedGraph[nonContractedNode].Color == boost::black_color)
                gomoryHuTree[newGomoryHuVertex].ContractedVertices.insert(nodeInSplitNode);
        }

        // Distribute the edges of the split node: Copy all edges adjacent to a black node to the
        // new node
        std::vector<TreeEdgeType> edgesToRemove;
        for (const auto e : boost::make_iterator_range(out_edges(splitNode, gomoryHuTree)))
        {
            const auto targetNode = target(e, gomoryHuTree);
            const auto sampleInputVertex = *begin(gomoryHuTree[targetNode].ContractedVertices);
            const auto sampleVertex = inputVertex2partiallyContractedMap.at(sampleInputVertex);

            if (partiallyContractedGraph[sampleVertex].Color == boost::black_color)
            {
                const auto weight = get(boost::edge_weight, gomoryHuTree, e);
                const auto [copiedEdge, _] = add_edge(newGomoryHuVertex, targetNode, gomoryHuTree);
                put(boost::edge_weight, gomoryHuTree, copiedEdge, weight);
                edgesToRemove.push_back(e);
            }
        }

        // Remove those edges that were just copied over to the new node
        // It's not safe to remove them in the previous loop (while iterating them)
        for (const auto e : edgesToRemove)
            remove_edge(e, gomoryHuTree);

        // Remove the black nodes from the split node that were just copied over to the new node
        std::erase_if(
            gomoryHuTree[splitNode].ContractedVertices,
            [&](const auto nodeInSplitNode)
            {
                const auto nonContractedNode
                    = inputVertex2partiallyContractedMap.at(nodeInSplitNode);
                return partiallyContractedGraph[nonContractedNode].Color == boost::black_color;
            });

        // most important step: connect the now fully split nodes with the cut size
        add_edge(newGomoryHuVertex, splitNode, cutSize, gomoryHuTree);

        if (gomoryHuTree[splitNode].ContractedVertices.size() > 1)
            treeNodesToBeSplit.push_back(splitNode);
        if (gomoryHuTree[newGomoryHuVertex].ContractedVertices.size() > 1)
            treeNodesToBeSplit.push_back(newGomoryHuVertex);
    }

    UndirectedGraph resultGraph(N);
    for (const auto treeEdge : boost::make_iterator_range(edges(gomoryHuTree)))
    {
        const auto s = source(treeEdge, gomoryHuTree);
        const auto t = target(treeEdge, gomoryHuTree);
        const auto weight = get(boost::edge_weight, gomoryHuTree, treeEdge);

        assert(gomoryHuTree[s].ContractedVertices.size() == 1);
        assert(gomoryHuTree[t].ContractedVertices.size() == 1);

        add_edge(
            *begin(gomoryHuTree[s].ContractedVertices), *begin(gomoryHuTree[t].ContractedVertices),
            weight, resultGraph);
    }

    return resultGraph;
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
