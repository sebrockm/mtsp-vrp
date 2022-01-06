#include "SupportGraphs.hpp"

#include <boost/graph/adjacency_matrix.hpp>
#include <boost/graph/boykov_kolmogorov_max_flow.hpp>
#include <boost/graph/filtered_graph.hpp>
#include <boost/property_map/function_property_map.hpp>

#include <xtensor/xvectorize.hpp>
#include <xtensor/xview.hpp>

tsplp::graph::PiSigmaSupportGraph::PiSigmaSupportGraph(const xt::xtensor<Variable, 3>& variables, const xt::xtensor<int, 2>& weights)
    : m_graph(variables.shape(1)), m_variables(variables), m_weights(weights)
{
    const auto N = variables.shape(1);
    assert(variables.shape(2) == N);
    assert(weights.shape(0) == N);
    assert(weights.shape(1) == N);

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
        const xt::xtensor<int, 2>* pw;
        ConstraintType x;
        PiSigmaVertex s;
        PiSigmaVertex t;

        bool operator()(PiSigmaVertex v) const
        {
            switch (x)
            {
            case ConstraintType::Pi: return (*pw)(s, v) != -1;
            case ConstraintType::Sigma: return (*pw)(v, t) != -1;
            case ConstraintType::PiSigma: return (*pw)(s, v) != -1 && (*pw)(v, t) != -1;
            }
        }
    };

    auto filteredSupportGraph = make_filtered_graph(m_graph, boost::keep_all{}, Filter{ &m_weights, x, s, t });

    const auto getCapacity = [&](PiSigmaEdge e)
    {
        // const auto vf = xt::vectorize([](Variable v) { return v.GetObjectiveValue(); });
        // const auto values = xt::view(m_variables, xt::all(), source(e, filteredSupportGraph), target(e, filteredSupportGraph));
        // return xt::sum(vf(values))();
        
        const auto A = m_variables.shape(0);

        // it turns out that this manual summing is much faster than the outcommented "nicer" version above
        double sum = 0.0;
        for (size_t a = 0; a < A; ++a)
            sum += m_variables(a, source(e, filteredSupportGraph), target(e, filteredSupportGraph)).GetObjectiveValue();
        return sum;
    };

    const auto getReverseEdge = [&](PiSigmaEdge e)
    {
        return edge(target(e, filteredSupportGraph), source(e, filteredSupportGraph), filteredSupportGraph).first;
    };

    const auto cutSize = boost::boykov_kolmogorov_max_flow(filteredSupportGraph,
        boost::make_function_property_map<PiSigmaEdge>(getCapacity),
        get(boost::edge_residual_capacity, filteredSupportGraph),
        boost::make_function_property_map<PiSigmaEdge>(getReverseEdge),
        get(boost::vertex_predecessor, filteredSupportGraph),
        get(boost::vertex_color, filteredSupportGraph),
        get(boost::vertex_distance, filteredSupportGraph),
        get(boost::vertex_index, filteredSupportGraph),
        s, t);

    if (cutSize >= 1.0 - 1.e-10)
        return { cutSize, {} };

    const auto colorMap = get(boost::vertex_color, filteredSupportGraph);

    const auto black = get(colorMap, s);
    assert(black == boost::black_color);
    assert(get(colorMap, t) != black);

    std::vector<std::pair<PiSigmaVertex, PiSigmaVertex>> result;
    result.reserve(num_edges(filteredSupportGraph));
    for (const auto edge : boost::make_iterator_range(edges(filteredSupportGraph)))
    {
        const auto u = source(edge, filteredSupportGraph);
        const auto v = target(edge, filteredSupportGraph);
        if (get(colorMap, u) == boost::black_color && get(colorMap, v) != boost::black_color)
            result.emplace_back(u, v);
    }

    return { cutSize, result };
}
