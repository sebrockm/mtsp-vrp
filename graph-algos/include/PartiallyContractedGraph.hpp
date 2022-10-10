#pragma once

#include <boost/graph/graph_traits.hpp>
#include <boost/graph/properties.hpp>
#include <boost/iterator/counting_iterator.hpp>
#include <boost/iterator/filter_iterator.hpp>
#include <boost/iterator/transform_iterator.hpp>
#include <boost/property_map/function_property_map.hpp>
#include <boost/property_map/property_map.hpp>

#include <utility>
#include <vector>

namespace graph_algos
{

struct PartiallyContractedGraph
{
    // graph concept
    using vertex_descriptor = size_t;
    using edge_descriptor = std::pair<size_t, size_t>;
    using directed_category = boost::directed_tag;
    using edge_parallel_category = boost::disallow_parallel_edge_tag;
    struct traversal_category : boost::vertex_list_graph_tag,
                                boost::edge_list_graph_tag,
                                boost::incidence_graph_tag,
                                boost::adjacency_matrix_tag
    {
    };

    static vertex_descriptor null_vertex() { return std::numeric_limits<vertex_descriptor>::max(); }

    // incidence graph concept
    struct FilterSame
    {
        vertex_descriptor u;
        bool operator()(vertex_descriptor v) const { return u != v; }
    };
    struct ToOutEdge
    {
        vertex_descriptor u;
        edge_descriptor operator()(vertex_descriptor v) const { return { u, v }; }
    };
    using out_edge_iterator = boost::transform_iterator<
        ToOutEdge, boost::filter_iterator<FilterSame, boost::counting_iterator<vertex_descriptor>>,
        edge_descriptor, edge_descriptor>;
    using degree_size_type = size_t;

    // vertex list graph concept
    using vertex_iterator = boost::counting_iterator<vertex_descriptor>;
    using vertices_size_type = size_t;

    // edge list graph concept
    struct N2Edge
    {
        size_t N;
        edge_descriptor operator()(size_t i) const
        {
            const auto u = i / (N - 1);
            const auto v = i % (N - 1);

            return { u, v + static_cast<size_t>(v >= u) };
        }
    };
    using edge_iterator = boost::transform_iterator<N2Edge, boost::counting_iterator<size_t>>;
    using edges_size_type = size_t;

    PartiallyContractedGraph(size_t maxN)
        : EdgeCapacities(maxN * (maxN - 1))
        , EdgeResidualCapacities(maxN * (maxN - 1))
        , VertexPredecessors(maxN)
        , VertexColors(maxN)
        , VertexDistances(maxN)
    {
    }

    void Reset(size_t n)
    {
        N = n;
        for (size_t i = 0; i < n * (n - 1); ++i)
            EdgeCapacities[i] = 0;
    }

    size_t N = 0;

    std::vector<double> EdgeCapacities;
    std::vector<double> EdgeResidualCapacities;

    std::vector<edge_descriptor> VertexPredecessors;
    std::vector<boost::default_color_type> VertexColors;
    std::vector<double> VertexDistances;
};

// incidence graph concept

auto source(const PartiallyContractedGraph::edge_descriptor& e, const PartiallyContractedGraph&)
{
    return e.first;
}

auto target(const PartiallyContractedGraph::edge_descriptor& e, const PartiallyContractedGraph&)
{
    return e.second;
}

auto out_edges(PartiallyContractedGraph::vertex_descriptor u, const PartiallyContractedGraph& g)
{
    PartiallyContractedGraph::FilterSame filter { u };
    PartiallyContractedGraph::ToOutEdge toOutEdge { u };
    return std::make_pair(
        PartiallyContractedGraph::out_edge_iterator({ filter, { 0 } }, toOutEdge),
        PartiallyContractedGraph::out_edge_iterator({ filter, { g.N } }, toOutEdge));
}

auto out_degree(PartiallyContractedGraph::vertex_descriptor, const PartiallyContractedGraph& g)
{
    return g.N - 1;
}

// vertex list graph concept

auto vertices(const PartiallyContractedGraph& g)
{
    return std::make_pair(
        PartiallyContractedGraph::vertex_iterator { 0 },
        PartiallyContractedGraph::vertex_iterator { g.N });
}

auto num_vertices(const PartiallyContractedGraph& g) { return g.N; }

// edge list graph concept

auto num_edges(const PartiallyContractedGraph& g) { return g.N * (g.N - 1); }

auto edges(const PartiallyContractedGraph& g)
{
    return std::make_pair(
        PartiallyContractedGraph::edge_iterator({ 0 }, { g.N }),
        PartiallyContractedGraph::edge_iterator({ num_edges(g) }, { g.N }));
}

// adjacency matrix graph concept

auto edge(
    PartiallyContractedGraph::vertex_descriptor u, PartiallyContractedGraph::vertex_descriptor v,
    const PartiallyContractedGraph&)
{
    return std::make_pair(PartiallyContractedGraph::edge_descriptor { u, v }, u != v);
}

// property maps for boykov_kolmogorov_max_flow

auto get(boost::edge_index_t, const PartiallyContractedGraph& g)
{
    return boost::make_function_property_map<PartiallyContractedGraph::edge_descriptor>(
        [N = g.N](PartiallyContractedGraph::edge_descriptor e)
        {
            const auto [u, v] = e;
            return (N - 1) * u + v - static_cast<size_t>(v >= u);
        });
}

auto get(boost::edge_capacity_t, PartiallyContractedGraph& g)
{
    return boost::make_iterator_property_map(g.EdgeCapacities.begin(), get(boost::edge_index, g));
}

auto get(boost::edge_residual_capacity_t, PartiallyContractedGraph& g)
{
    return boost::make_iterator_property_map(
        g.EdgeResidualCapacities.begin(), get(boost::edge_index, g));
}

auto get(boost::edge_reverse_t, const PartiallyContractedGraph&)
{
    return boost::make_function_property_map<PartiallyContractedGraph::edge_descriptor>(
        [](PartiallyContractedGraph::edge_descriptor e) {
            return PartiallyContractedGraph::edge_descriptor { e.second, e.first };
        });
}

auto get(boost::vertex_index_t, const PartiallyContractedGraph&)
{
    return boost::typed_identity_property_map<PartiallyContractedGraph::vertex_descriptor> {};
}

auto get(boost::vertex_predecessor_t, PartiallyContractedGraph& g)
{
    return boost::make_container_vertex_map(g.VertexPredecessors);
}

auto get(boost::vertex_color_t, PartiallyContractedGraph& g)
{
    return boost::make_container_vertex_map(g.VertexColors);
}

auto get(boost::vertex_distance_t, PartiallyContractedGraph& g)
{
    return boost::make_container_vertex_map(g.VertexDistances);
}

}
