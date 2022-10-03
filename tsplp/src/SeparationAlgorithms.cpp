#include "SeparationAlgorithms.hpp"

#include "GomoryHuTree.hpp"
#include "LinearConstraint.hpp"
#include "LinearVariableComposition.hpp"
#include "Model.hpp"
#include "SupportGraphs.hpp"
#include "Variable.hpp"
#include "WeightManager.hpp"

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

std::optional<LinearConstraint> Separator::TwoMatching() const
{
    const auto N = m_weightManager.N();
    const auto vf = xt::vectorize([this](Variable v) { return v.GetObjectiveValue(m_model); });
    const auto values = vf(m_variables);

    UndirectedGraph graph(N);
    struct hash
    {
        size_t operator()(const EdgeType& e) const { return boost::core::bit_cast<size_t>(e.m_eproperty); }
    };
    std::unordered_map<EdgeType, double, hash> edge2WeightMap;
    for (size_t u = 0; u < N; ++u)
    {
        for (size_t v = u + 1; v < N; ++v)
        {
            const auto weight = std::max(
                0.0,
                std::min(
                    1.0,
                    xt::sum(xt::view(values, xt::all(), u, v))()
                        + xt::sum(xt::view(values, xt::all(), v, u))()));
            const auto capacity = std::min(weight, 1 - weight);
            const auto [edge, inserted] = boost::add_edge(u, v, capacity, graph);
            assert(inserted);
            edge2WeightMap.emplace(edge, weight);
        }
    }

    std::vector<bool> odd(N);
    for (const auto v : boost::make_iterator_range(vertices(graph)))
    {
        const auto [eBegin, eEnd] = out_edges(v, graph);
        odd[v] = std::count_if(
                     eBegin, eEnd, [&](const EdgeType& e) { return edge2WeightMap.at(e) > 0.5; })
                % 2
            == 1;
    }

    std::optional<LinearConstraint> result {};

    size_t counter = 0;

    CreateGomoryHuTree(
        graph,
        [&](const double cutSize, std::span<const size_t> comp1, std::span<const size_t> comp2)
        {
            ++counter;
            assert(cutSize >= 0);

            const auto ForAllCutEdges = [&](auto f)
            {
                for (const auto u : comp1)
                {
                    for (const auto v : comp2)
                    {
                        f(u, v);
                    }
                }
            };

            bool isOdd = false;
            for (const auto v : comp1)
                isOdd ^= odd[v];

            if (cutSize < 1 - 1e-10 && isOdd)
            {
                LinearVariableComposition lhs = 0;
                LinearVariableComposition rhs = 1;

                ForAllCutEdges(
                    [&](size_t u, size_t v)
                    {
                        if (const auto [edge, exists] = boost::edge(u, v, graph);
                            exists && edge2WeightMap.at(edge) > 0.5)
                        {
                            rhs += xt::sum(xt::view(m_variables, xt::all(), u, v) + 0)()
                                + xt::sum(xt::view(m_variables, xt::all(), v, u) + 0)() - 1;
                        }
                        else
                        {
                            lhs += xt::sum(xt::view(m_variables, xt::all(), u, v) + 0)()
                                + xt::sum(xt::view(m_variables, xt::all(), v, u) + 0)();
                        }
                    });

                result = lhs >= rhs;
                return true;
            }
            else
            {
                double w1 = 1;
                double w2 = 0;
                EdgeType e1 {};
                EdgeType e2 {};
                ForAllCutEdges(
                    [&](size_t u, size_t v)
                    {
                        if (const auto [edge, exists] = boost::edge(u, v, graph);
                            exists && edge2WeightMap.at(edge) > 0.5)
                        {
                            w1 = std::min(w1, edge2WeightMap.at(edge));
                            e1 = edge;
                        }
                        else
                        {
                            w2 = std::max(w2, edge2WeightMap.at(edge));
                            e2 = edge;
                        }
                    });

                if (cutSize + std::min(2 * w1 - 1, 1 - 2 * w2) < 1 - 1e-10)
                {
                    LinearVariableComposition lhs = 0;
                    LinearVariableComposition rhs = 1;

                    ForAllCutEdges(
                        [&](size_t u, size_t v)
                        {
                            if (const auto [edge, exists] = boost::edge(u, v, graph); exists
                                && ((edge2WeightMap.at(edge) > 0.5
                                     || (2 * w1 - 1 >= 1 - 2 * w2 && edge == e2))
                                    || (edge2WeightMap.at(edge) > 0.5 && 2 * w1 - 1 < 1 - 2 * w2
                                        && edge != e1)))
                            {
                                rhs += xt::sum(xt::view(m_variables, xt::all(), u, v) + 0)()
                                    + xt::sum(xt::view(m_variables, xt::all(), v, u) + 0)() - 1;
                            }
                            else
                            {
                                lhs += xt::sum(xt::view(m_variables, xt::all(), u, v) + 0)()
                                    + xt::sum(xt::view(m_variables, xt::all(), v, u) + 0)();
                            }
                        });

                    result = lhs >= rhs;
                    return true;
                }
            }

            return false;
        });

    //std::cout << "comb " << (result ? "" : "not") << " found after " << counter << " of " << N - 1
    //          << "gomory hu tree edges were found" << std::endl;

    return result;
}
}
