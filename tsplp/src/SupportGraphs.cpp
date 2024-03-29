#include "SupportGraphs.hpp"

#include <boost/graph/boykov_kolmogorov_max_flow.hpp>
#include <boost/graph/filtered_graph.hpp>
#include <boost/property_map/function_property_map.hpp>
#include <xtensor/xvectorize.hpp>
#include <xtensor/xview.hpp>

tsplp::graph::PiSigmaSupportGraph::PiSigmaSupportGraph(
    const xt::xtensor<Variable, 3>& variables, const DependencyGraph& dependencies,
    const Model& model)
    : m_graph(variables.shape(1))
    , m_variables(variables)
    , m_dependencies(dependencies)
    , m_model(model)
{
    const auto N = variables.shape(1);
    assert(variables.shape(2) == N);

    for (size_t u = 0; u < N; ++u)
    {
        for (size_t v = 0; v < N; ++v)
        {
            if (u != v)
                add_edge(u, v, m_graph);
        }
    }

    assert(num_vertices(m_graph) == N);
    assert(num_edges(m_graph) == N * N - N);
}

std::pair<double, std::vector<std::pair<tsplp::graph::PiSigmaVertex, tsplp::graph::PiSigmaVertex>>>
tsplp::graph::PiSigmaSupportGraph::FindMinCut(PiSigmaVertex s, PiSigmaVertex t, ConstraintType x)
{
    struct Filter // cannot be a lambda because the filter must be default constructible
    {
        const DependencyGraph* d;
        ConstraintType x;
        PiSigmaVertex s;
        PiSigmaVertex t;

        bool operator()(PiSigmaVertex v) const
        {
            switch (x)
            {
            case ConstraintType::Pi:
                return !d->HasArc(v, s);
            case ConstraintType::Sigma:
                return !d->HasArc(t, v);
            case ConstraintType::PiSigma:
                return !d->HasArc(v, s) && !d->HasArc(t, v);
            }
            return true;
        }
    };

    auto filteredSupportGraph
        = make_filtered_graph(m_graph, boost::keep_all {}, Filter { &m_dependencies, x, s, t });

    const auto getCapacity = [&](const PiSigmaEdge& e)
    {
        const auto A = m_variables.shape(0);

        double sum = 0.0;
        for (size_t a = 0; a < A; ++a)
        {
            sum += m_variables(a, source(e, filteredSupportGraph), target(e, filteredSupportGraph))
                       .GetObjectiveValue(m_model);
        }
        return sum;
    };

    const auto getReverseEdge = [&](const PiSigmaEdge& e)
    {
        return edge(
                   target(e, filteredSupportGraph), source(e, filteredSupportGraph),
                   filteredSupportGraph)
            .first;
    };

    const auto cutSize = boost::boykov_kolmogorov_max_flow(
        filteredSupportGraph, boost::make_function_property_map<PiSigmaEdge>(getCapacity),
        get(boost::edge_residual_capacity, filteredSupportGraph),
        boost::make_function_property_map<PiSigmaEdge>(getReverseEdge),
        get(boost::vertex_predecessor, filteredSupportGraph),
        get(boost::vertex_color, filteredSupportGraph),
        get(boost::vertex_distance, filteredSupportGraph),
        get(boost::vertex_index, filteredSupportGraph), s, t);

    if (cutSize >= 1.0 - 1.e-10)
        return { cutSize, {} };

    const auto colorMap = get(boost::vertex_color, filteredSupportGraph);

    [[maybe_unused]] const auto black = get(colorMap, s);
    assert(black == boost::black_color);
    assert(get(colorMap, t) != black);

    std::vector<std::pair<PiSigmaVertex, PiSigmaVertex>> result;
    result.reserve(num_edges(filteredSupportGraph));
    for (const auto& edge : boost::make_iterator_range(edges(filteredSupportGraph)))
    {
        const auto u = source(edge, filteredSupportGraph);
        const auto v = target(edge, filteredSupportGraph);
        if (get(colorMap, u) == boost::black_color && get(colorMap, v) != boost::black_color)
            result.emplace_back(u, v);
    }

    return { cutSize, result };
}
