#include "SeparationAlgorithms.hpp"

#include "LinearConstraint.hpp"
#include "LinearVariableComposition.hpp"
#include "Variable.hpp"
#include "WeightManager.hpp"

#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/one_bit_color_map.hpp>
#include <boost/graph/stoer_wagner_min_cut.hpp>
#include <boost/graph/filtered_graph.hpp>
#include <boost/graph/boykov_kolmogorov_max_flow.hpp>
#include <boost/range/iterator_range.hpp>

#include <xtensor/xtensor.hpp>
#include <xtensor/xvectorize.hpp>
#include <xtensor/xview.hpp>

namespace tsplp::graph
{
    using EdgeWeightProperty = boost::property<boost::edge_weight_t, double>;
    using UndirectedGraph = boost::adjacency_list<boost::vecS, boost::vecS, boost::undirectedS, boost::no_property, EdgeWeightProperty>;

    Separator::Separator(const xt::xtensor<Variable, 3>& variables, const WeightManager& weightManager)
        : m_variables(variables), m_weightManager(weightManager)
    {
    }

    std::optional<LinearConstraint> Separator::Ucut() const
    {
        const auto N = m_weightManager.N();
        const auto vf = xt::vectorize([](Variable v) { return v.GetObjectiveValue(); });
        const auto values = vf(m_variables);

        UndirectedGraph graph(N);
        for (size_t u = 0; u < N; ++u)
        {
            for (size_t v = u + 1; v < N; ++v)
            {
                const auto weight = xt::sum(xt::view(values, xt::all(), u, v))() + xt::sum(xt::view(values, xt::all(), v, u))();
                boost::add_edge(u, v, weight, graph);
            }
        }

        const auto parities = boost::make_one_bit_color_map(N, get(boost::vertex_index, graph));

        const auto cutSize = boost::stoer_wagner_min_cut(graph, get(boost::edge_weight, graph), boost::parity_map(parities));

        if (cutSize >= 2.0 - 1.e-10)
            return std::nullopt;

        LinearVariableComposition sum;
        for (size_t u = 0; u < N; ++u)
            for (size_t v = 0; v < N; ++v)
                if (get(parities, u) != get(parities, v))
                    sum += xt::sum(xt::view(m_variables, xt::all(), u, v) + 0)();

        return sum >= 2;
    }

    using PiSigmaGraphTraits = boost::adjacency_list_traits<boost::vecS, boost::vecS, boost::directedS>;
    using PiSigmaVertexProperties =
        boost::property<boost::vertex_predecessor_t, PiSigmaGraphTraits::edge_descriptor,
        boost::property<boost::vertex_color_t, boost::default_color_type,
        boost::property<boost::vertex_distance_t, double>>>;
    using PiSigmaEdgeProperties =
        boost::property<boost::edge_capacity_t, double,
        boost::property<boost::edge_residual_capacity_t, double,
        boost::property<boost::edge_reverse_t, PiSigmaGraphTraits::edge_descriptor>>>;
    using PiSigmaSupportGraph = boost::adjacency_list<boost::vecS, boost::vecS, boost::directedS, PiSigmaVertexProperties, PiSigmaEdgeProperties>;

    std::vector<LinearConstraint> Separator::Pi() const
    {
        const auto N = m_weightManager.N();
        const auto A = m_weightManager.A();
        const auto vf = xt::vectorize([](Variable v) { return v.GetObjectiveValue(); });
        const auto values = vf(m_variables);

        PiSigmaSupportGraph supportGraph(N);
        auto capacityMap = get(boost::edge_capacity, supportGraph);
        auto reverseEdgeMap = get(boost::edge_reverse, supportGraph);

        for (size_t u = 0; u < N; ++u)
        {
            for (size_t v = u + 1; v < N; ++v)
            {
                [[maybe_unused]] const auto [uv, unused1] = add_edge(u, v, supportGraph);
                [[maybe_unused]] const auto [vu, unused2] = add_edge(v, u, supportGraph);
                reverseEdgeMap[uv] = vu;
                reverseEdgeMap[vu] = uv;

                if (auto const xuv = xt::sum(xt::view(values, xt::all(), u, v))(); xuv > 1.e-10)
                    capacityMap[uv] = xuv;

                if (auto const xvu = xt::sum(xt::view(values, xt::all(), v, u))(); xvu > 1.e-10)
                    capacityMap[vu] = xvu;
            }
        }

        assert(num_vertices(supportGraph) == N);
        assert(num_edges(supportGraph) == N * N - N);

        std::vector<LinearConstraint> result;
        for (size_t n = 0; n < N; ++n)
        {
            if (std::find(m_weightManager.EndPositions().begin(), m_weightManager.EndPositions().end(), n) != m_weightManager.EndPositions().end())
                continue;

            const auto nDependsOn = equal(xt::view(m_weightManager.W(), n, xt::all()), -1);
            if (!xt::any(nDependsOn))
                continue;

            struct Filter
            {
                const decltype(nDependsOn)* m_p;
                bool operator()(PiSigmaGraphTraits::vertex_descriptor v) const { return !(*m_p)[v]; }
            };

            auto filteredSupportGraph = make_filtered_graph(supportGraph, boost::keep_all{}, Filter{ &nDependsOn });

            for (size_t a = 0; a < A; ++a)
            {
                const auto e = static_cast<size_t>(m_weightManager.EndPositions()[a]);

                const auto cutSize = boost::boykov_kolmogorov_max_flow(filteredSupportGraph, n, e);

                if (cutSize < 1.0 - 1.e-10)
                {
                    const auto colorMap = get(boost::vertex_color, filteredSupportGraph);

                    const auto black = get(colorMap, n);
                    assert(black == boost::black_color);
                    assert(get(colorMap, e) != black);

                    LinearVariableComposition sum;
                    for (const auto edge : boost::make_iterator_range(edges(filteredSupportGraph)))
                    {
                        const auto u = source(edge, filteredSupportGraph);
                        const auto v = target(edge, filteredSupportGraph);
                        if (get(colorMap, u) == black && get(colorMap, v) != black)
                            sum += xt::sum(xt::view(m_variables, xt::all(), u, v) + 0)();
                    }

                    auto constraint = sum >= 1;
                    assert(!constraint.Evaluate());
                    result.push_back(std::move(constraint));
                }
            }
        }

        return result;
    }

    std::vector<LinearConstraint> Separator::Sigma() const
    {
        static std::unordered_set<LinearConstraint> s_all;
        const auto N = m_weightManager.N();
        const auto A = m_weightManager.A();
        const auto vf = xt::vectorize([](Variable v) { return v.GetObjectiveValue(); });
        const auto values = vf(m_variables);

        PiSigmaSupportGraph supportGraph(N);
        auto capacityMap = get(boost::edge_capacity, supportGraph);
        auto reverseEdgeMap = get(boost::edge_reverse, supportGraph);

        for (size_t u = 0; u < N; ++u)
        {
            for (size_t v = u + 1; v < N; ++v)
            {
                [[maybe_unused]] const auto [uv, unused1] = add_edge(u, v, supportGraph);
                [[maybe_unused]] const auto [vu, unused2] = add_edge(v, u, supportGraph);
                reverseEdgeMap[uv] = vu;
                reverseEdgeMap[vu] = uv;

                if (auto const xuv = xt::sum(xt::view(values, xt::all(), u, v))(); xuv > 1.e-10)
                    capacityMap[uv] = xuv;

                if (auto const xvu = xt::sum(xt::view(values, xt::all(), v, u))(); xvu > 1.e-10)
                    capacityMap[vu] = xvu;
            }
        }

        assert(num_vertices(supportGraph) == N);
        assert(num_edges(supportGraph) == N * N - N);

        std::vector<LinearConstraint> result;

        for (size_t n = 0; n < N; ++n)
        {
            if (std::find(m_weightManager.StartPositions().begin(), m_weightManager.StartPositions().end(), n) != m_weightManager.StartPositions().end())
                continue;

            const auto dependsOnN = equal(xt::view(m_weightManager.W(), xt::all(), n), -1);
            if (!xt::any(dependsOnN))
                continue;

            struct Filter
            {
                const decltype(dependsOnN)* m_p;
                bool operator()(PiSigmaGraphTraits::vertex_descriptor v) const { return !(*m_p)[v]; }
            };

            auto filteredSupportGraph = make_filtered_graph(supportGraph, boost::keep_all{}, Filter{ &dependsOnN });

            for (size_t a = 0; a < A; ++a)
            {
                const auto s = static_cast<size_t>(m_weightManager.StartPositions()[a]);

                const auto cutSize = boost::boykov_kolmogorov_max_flow(filteredSupportGraph, s, n);

                if (cutSize < 1.0 - 1.e-10)
                {
                    const auto colorMap = get(boost::vertex_color, filteredSupportGraph);

                    const auto black = get(colorMap, s);
                    assert(black == boost::black_color);
                    assert(get(colorMap, n) != black);

                    LinearVariableComposition sum;
                    for (const auto edge : boost::make_iterator_range(edges(filteredSupportGraph)))
                    {
                        const auto u = source(edge, filteredSupportGraph);
                        const auto v = target(edge, filteredSupportGraph);
                        if (get(colorMap, u) == black && get(colorMap, v) != black)
                            sum += xt::sum(xt::view(m_variables, xt::all(), u, v) + 0)();
                    }

                    auto constraint = sum >= 1;
                    assert(!constraint.Evaluate());
                    result.push_back(std::move(constraint));
                }
            }
        }

        return result;
    }
}