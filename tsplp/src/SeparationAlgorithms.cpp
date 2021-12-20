#include "SeparationAlgorithms.hpp"

#include "LinearConstraint.hpp"
#include "LinearVariableComposition.hpp"
#include "Variable.hpp"

#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/one_bit_color_map.hpp>
#include <boost/graph/stoer_wagner_min_cut.hpp>

#include <xtensor/xtensor.hpp>
#include <xtensor/xvectorize.hpp>
#include <xtensor/xview.hpp>

namespace tsplp::graph
{
    using EdgeWeightProperty = boost::property<boost::edge_weight_t, double>;
    using UndirectedGraph = boost::adjacency_list<boost::vecS, boost::vecS, boost::undirectedS, boost::no_property, EdgeWeightProperty>;

    Separator::Separator(const xt::xtensor<Variable, 3>& variables)
        : m_variables(variables)
    {
    }

    std::optional<LinearConstraint> Separator::Ucut() const
    {
        const auto N = m_variables.shape(1);
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
}