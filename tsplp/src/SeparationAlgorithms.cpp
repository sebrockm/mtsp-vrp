#include "SeparationAlgorithms.hpp"

#include "LinearConstraint.hpp"
#include "LinearVariableComposition.hpp"
#include "Model.hpp"
#include "SupportGraphs.hpp"
#include "Variable.hpp"
#include "WeightManager.hpp"

#include <GomoryHuTree.hpp>

#include <boost/core/bit.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/one_bit_color_map.hpp>
#include <boost/graph/stoer_wagner_min_cut.hpp>
#include <boost/range/iterator_range.hpp>
#include <xtensor/xtensor.hpp>
#include <xtensor/xvectorize.hpp>
#include <xtensor/xview.hpp>

namespace tsplp::graph
{
using EdgeWeightProperty = boost::property<boost::edge_weight_t, double>;
using UndirectedGraph = boost::adjacency_list<
    boost::vecS, boost::vecS, boost::undirectedS, boost::no_property, EdgeWeightProperty>;

Separator::Separator(
    const xt::xtensor<Variable, 3>& variables, const WeightManager& weightManager,
    const Model& model)
    : m_variables(variables)
    , m_weightManager(weightManager)
    , m_model(model)
    , m_spSupportGraph( // TODO: create only on demand
          std::make_unique<PiSigmaSupportGraph>(variables, weightManager.Dependencies(), model))
{
}

Separator::~Separator() noexcept = default;

std::optional<LinearConstraint> Separator::Ucut() const
{
    const auto N = m_weightManager.N();
    const auto vf = xt::vectorize([this](Variable v) { return v.GetObjectiveValue(m_model); });
    const auto values = vf(m_variables);

    UndirectedGraph graph(N);
    for (size_t u = 0; u < N; ++u)
    {
        for (size_t v = u + 1; v < N; ++v)
        {
            const auto weight = xt::sum(xt::view(values, xt::all(), u, v))()
                + xt::sum(xt::view(values, xt::all(), v, u))();
            boost::add_edge(u, v, weight, graph);
        }
    }

    const auto parities = boost::make_one_bit_color_map(N, get(boost::vertex_index, graph));

    const auto cutSize = boost::stoer_wagner_min_cut(
        graph, get(boost::edge_weight, graph), boost::parity_map(parities));

    if (cutSize >= 2.0 - 1.e-10)
        return std::nullopt;

    LinearVariableComposition sum;
    for (size_t u = 0; u < N; ++u)
        for (size_t v = 0; v < N; ++v)
            if (get(parities, u) != get(parities, v))
                sum += xt::sum(xt::view(m_variables, xt::all(), u, v) + 0)();

    return sum >= 2;
}

std::optional<LinearConstraint> Separator::Pi() const
{
    if (m_weightManager.Dependencies().GetArcs().empty())
        return std::nullopt;

    const auto N = m_weightManager.N();
    const auto A = m_weightManager.A();

    for (size_t n = 0; n < N; ++n)
    {
        if (m_weightManager.Dependencies().GetIncomingSpan(n).empty())
            continue;

        for (size_t a = 0; a < A; ++a)
        {
            const auto e = m_weightManager.EndPositions()[a];
            if (n == e)
                continue;

            const auto [cutSize, cutEdges]
                = m_spSupportGraph->FindMinCut(n, e, PiSigmaSupportGraph::ConstraintType::Pi);

            if (cutSize < 1.0 - 1.e-10)
            {

                LinearVariableComposition sum;
                for (const auto& [u, v] : cutEdges)
                    sum += xt::sum(xt::view(m_variables, xt::all(), u, v) + 0)();

                assert(std::abs(sum.Evaluate(m_model) - cutSize) < 1.e-10);
                auto constraint = sum >= 1;
                assert(!constraint.Evaluate(m_model));

                return constraint;
            }
        }
    }

    return std::nullopt;
}

std::optional<LinearConstraint> Separator::Sigma() const
{
    if (m_weightManager.Dependencies().GetArcs().empty())
        return std::nullopt;

    const auto N = m_weightManager.N();
    const auto A = m_weightManager.A();

    for (size_t n = 0; n < N; ++n)
    {
        if (m_weightManager.Dependencies().GetOutgoingSpan(n).empty())
            continue;

        for (size_t a = 0; a < A; ++a)
        {
            const auto s = m_weightManager.StartPositions()[a];
            if (n == s)
                continue;

            const auto [cutSize, cutEdges]
                = m_spSupportGraph->FindMinCut(s, n, PiSigmaSupportGraph::ConstraintType::Sigma);

            if (cutSize < 1.0 - 1.e-10)
            {
                LinearVariableComposition sum;
                for (const auto& [u, v] : cutEdges)
                    sum += xt::sum(xt::view(m_variables, xt::all(), u, v) + 0)();

                auto constraint = sum >= 1;
                assert(!constraint.Evaluate(m_model));

                return constraint;
            }
        }
    }

    return std::nullopt;
}

std::optional<LinearConstraint> Separator::PiSigma() const
{
    if (m_weightManager.Dependencies().GetArcs().empty())
        return std::nullopt;

    for (const auto& [s, t] : m_weightManager.Dependencies().GetArcs())
    {
        const auto [cutSize, cutEdges]
            = m_spSupportGraph->FindMinCut(s, t, PiSigmaSupportGraph::ConstraintType::PiSigma);

        if (cutSize < 1.0 - 1.e-10)
        {
            LinearVariableComposition sum;
            for (const auto& [u, v] : cutEdges)
                sum += xt::sum(xt::view(m_variables, xt::all(), u, v) + 0)();

            assert(std::abs(sum.Evaluate(m_model) - cutSize) < 1.e-10);
            auto constraint = sum >= 1;
            assert(!constraint.Evaluate(m_model));

            return constraint;
        }
    }

    return std::nullopt;
}

std::vector<LinearConstraint> Separator::TwoMatching() const
{
    const auto N = m_weightManager.N();
    const auto vf = xt::vectorize([this](Variable v) { return v.GetObjectiveValue(m_model); });
    const auto values = vf(m_variables);

    graph_algos::UndirectedGraph graph(N);
    std::vector<double> edge2WeightMap(N * (N - 1) / 2);
    for (size_t u = 0; u < N; ++u)
    {
        for (size_t v = 0; v < u; ++v)
        {
            const auto weight = std::max(
                0.0,
                std::min(
                    1.0,
                    xt::sum(xt::view(values, xt::all(), u, v))()
                        + xt::sum(xt::view(values, xt::all(), v, u))()));
            const auto capacity = std::min(weight, 1 - weight);
            boost::add_edge(u, v, capacity, graph);
            edge2WeightMap[u * (u - 1) / 2 + v] = weight;
        }
    }
    const auto edge2WeightFunction = [&](const graph_algos::EdgeType& e)
    {
        const auto s = source(e, graph);
        const auto t = target(e, graph);
        const auto u = std::max(s, t);
        const auto v = std::min(s, t);
        return edge2WeightMap[u * (u - 1) / 2 + v];
    };

    std::vector<bool> odd(N);
    for (const auto v : boost::make_iterator_range(vertices(graph)))
    {
        const auto [eBegin, eEnd] = out_edges(v, graph);
        odd[v] = std::count_if(
                     eBegin, eEnd,
                     [&](const graph_algos::EdgeType& e) { return edge2WeightFunction(e) > 0.5; })
                % 2
            == 1;
    }

    const auto IsOdd = [&](const std::vector<size_t>& componentIds)
    {
        bool isOdd = false;
        for (size_t v = 0; v < N; ++v)
            isOdd ^= componentIds[v] == 0 && odd[v];
        return isOdd;
    };

    std::vector<LinearConstraint> results {};

    const auto gomoryHuTree = graph_algos::CreateGomoryHuTree(graph);
    for (const auto& e : boost::make_iterator_range(edges(gomoryHuTree)))
    {
        const auto cutSize = get(boost::edge_weight, gomoryHuTree, e);
        assert(cutSize >= 0);

        if (cutSize >= 1 - 1e-10)
            continue;

        struct Filter
        {
            graph_algos::EdgeType edge;
            bool operator()(const graph_algos::EdgeType& e) const { return edge != e; }
        };

        std::vector<size_t> componentIds(N);
        [[maybe_unused]] const auto numberOfComponents = boost::connected_components(
            make_filtered_graph(gomoryHuTree, Filter { e }, boost::keep_all {}),
            componentIds.data());

        assert(numberOfComponents == 2);

        const auto ForAllCutEdges = [&](auto f)
        {
            for (size_t u = 0; u < N; ++u)
            {
                if (componentIds[u] != 0)
                    continue;

                for (size_t v = 0; v < u; ++v)
                {
                    if (componentIds[v] != 1)
                        continue;

                    f(u, v);
                }
            }
        };

        if (cutSize < 1 - 1e-10 && IsOdd(componentIds))
        {
            LinearVariableComposition lhs = 0;
            LinearVariableComposition rhs = 1;

            ForAllCutEdges(
                [&](size_t u, size_t v)
                {
                    auto constraintPart = xt::sum(xt::view(m_variables, xt::all(), u, v) + 0)()
                        + xt::sum(xt::view(m_variables, xt::all(), v, u) + 0)();
                    if (edge2WeightMap[u * (u - 1) / 2 + v] > 0.5)
                    {
                        rhs += std::move(constraintPart) - 1;
                    }
                    else
                    {
                        lhs += std::move(constraintPart);
                    }
                });

            results.push_back(std::move(lhs) >= std::move(rhs));
        }
        else
        {
            double w1 = 1;
            double w2 = 0;
            std::pair<size_t, size_t> e1 {};
            std::pair<size_t, size_t> e2 {};
            ForAllCutEdges(
                [&](size_t u, size_t v)
                {
                    if (const auto weight = edge2WeightMap[u * (u - 1) / 2 + v]; weight > 0.5)
                    {
                        w1 = std::min(w1, weight);
                        e1 = { u, v };
                    }
                    else
                    {
                        w2 = std::max(w2, weight);
                        e2 = { u, v };
                    }
                });

            if (cutSize + std::min(2 * w1 - 1, 1 - 2 * w2) < 1 - 1e-10)
            {
                LinearVariableComposition lhs = 0;
                LinearVariableComposition rhs = 1;

                ForAllCutEdges(
                    [&](size_t u, size_t v)
                    {
                        auto constraintPart = xt::sum(xt::view(m_variables, xt::all(), u, v) + 0)()
                            + xt::sum(xt::view(m_variables, xt::all(), v, u) + 0)();

                        const auto weight = edge2WeightMap[u * (u - 1) / 2 + v];
                        const auto edge = std::make_pair(u, v);

                        if ((2 * w1 - 1 < 1 - 2 * w2 && weight > 0.5 && edge != e1)
                            || (2 * w1 - 1 >= 1 - 2 * w2 && (weight > 0.5 || edge == e2)))
                        {
                            rhs += std::move(constraintPart) - 1;
                        }
                        else
                        {
                            lhs += std::move(constraintPart);
                        }
                    });

                results.push_back(std::move(lhs) >= std::move(rhs));
            }
        }
    }

    return results;
}
}
