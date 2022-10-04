#pragma once

#include <boost/graph/adjacency_list.hpp>

namespace graph_algos
{

using EdgeWeightProperty = boost::property<boost::edge_weight_t, double>;
using UndirectedGraph = boost::adjacency_list<
    boost::vecS, boost::vecS, boost::undirectedS, boost::no_property, EdgeWeightProperty>;
using VertexType = typename boost::graph_traits<UndirectedGraph>::vertex_descriptor;
using EdgeType = typename boost::graph_traits<UndirectedGraph>::edge_descriptor;

UndirectedGraph CreateGomoryHuTree(const UndirectedGraph& inputGraph);

double GetMinCutFromGomoryHuTree(
    const UndirectedGraph& gomoryHuTree, VertexType source, VertexType sink);

}
