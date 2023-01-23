#pragma once

#include <boost/graph/depth_first_search.hpp>
#include <boost/graph/graph_traits.hpp>
#include <boost/graph/undirected_dfs.hpp>
#include <boost/property_map/property_map.hpp>

#include <unordered_map>

namespace graph_algos
{
struct CycleDetector : public boost::dfs_visitor<>
{
    void back_edge(auto, auto) { throw 0; }
};

template <typename Graph>
[[nodiscard]] bool HasCycle(const Graph& graph)
{
    try
    {
        if constexpr (std::is_same_v<
                          boost::directed_tag,
                          typename boost::graph_traits<Graph>::directed_category>)
        {
            boost::depth_first_search(graph, visitor(CycleDetector {}));
        }
        else
        {
            using Edge = typename boost::graph_traits<Graph>::edge_descriptor;
            std::unordered_map<Edge, boost::default_color_type, boost::hash<Edge>> edgeColorMap;
            const auto edge_color_map = boost::make_assoc_property_map(edgeColorMap);

            boost::undirected_dfs(graph, visitor(CycleDetector {}).edge_color_map(edge_color_map));
        }
    }
    catch (int)
    {
        return true;
    }

    return false;
}
}
