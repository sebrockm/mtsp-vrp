#pragma once

#include <boost/graph/adjacency_list.hpp>

#include <functional>

namespace tsplp::graph
{

using EdgeWeightProperty = boost::property<boost::edge_weight_t, double>;
using UndirectedGraph = boost::adjacency_list<
    boost::vecS, boost::vecS, boost::undirectedS, boost::no_property, EdgeWeightProperty>;
using VertexType = typename boost::graph_traits<UndirectedGraph>::vertex_descriptor;
using EdgeType = typename boost::graph_traits<UndirectedGraph>::edge_descriptor;

void CreateGomoryHuTree(
    const UndirectedGraph& inputGraph,
    std::function<
        bool(double cutSize, std::vector<VertexType> comp1, std::vector<VertexType> comp2)>
        newEdgeCallback);

double GetMinCutFromGomoryHuTree(
    const UndirectedGraph& gomoryHuTree, VertexType source, VertexType sink);

}