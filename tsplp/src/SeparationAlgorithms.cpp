#include "SeparationAlgorithms.hpp"

#include "LinearConstraint.hpp"
#include "LinearVariableComposition.hpp"
#include "Model.hpp"
#include "SupportGraphs.hpp"
#include "Variable.hpp"
#include "WeightManager.hpp"

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
    using UndirectedGraph = boost::adjacency_list<boost::vecS, boost::vecS, boost::undirectedS, boost::no_property, EdgeWeightProperty>;

    Separator::Separator(const xt::xtensor<Variable, 3>& variables, const WeightManager& weightManager, const Model& model)
        : m_variables(variables), m_weightManager(weightManager), m_model(model), m_spSupportGraph(std::make_unique<PiSigmaSupportGraph>(variables, weightManager.Dependencies(), model))
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

                const auto [cutSize, cutEdges] = m_spSupportGraph->FindMinCut(n, e, PiSigmaSupportGraph::ConstraintType::Pi);

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

                const auto [cutSize, cutEdges] = m_spSupportGraph->FindMinCut(s, n, PiSigmaSupportGraph::ConstraintType::Sigma);

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
            const auto [cutSize, cutEdges] = m_spSupportGraph->FindMinCut(s, t, PiSigmaSupportGraph::ConstraintType::PiSigma);

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
}
