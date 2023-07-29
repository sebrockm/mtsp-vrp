#pragma once

#include <boost/graph/adjacency_list.hpp>

#include <functional>
#include <span>

namespace graph_algos
{

using EdgeWeightProperty = boost::property<boost::edge_weight_t, double>;
using UndirectedGraph = boost::adjacency_list<
    boost::vecS, boost::vecS, boost::undirectedS, boost::no_property, boost::no_property>;
using VertexType = typename boost::graph_traits<UndirectedGraph>::vertex_descriptor;
using EdgeType = typename boost::graph_traits<UndirectedGraph>::edge_descriptor;

void CreateGomoryHuTree(
    size_t N, std::vector<double> const& edge2CapacityMap,
    const std::function<bool(
        VertexType u, VertexType v, double cutSize, std::span<const VertexType> compU,
        std::span<const VertexType> compV)>& newEdgeCallback);

double GetMinCutFromGomoryHuTree(
    const UndirectedGraph& gomoryHuTree, VertexType source, VertexType sink);

}
