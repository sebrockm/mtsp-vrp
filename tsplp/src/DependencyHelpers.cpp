#include "DependencyHelpers.hpp"

#include <ranges>

#include <boost/graph/transitive_closure.hpp>

namespace tsplp
{
    xt::xtensor<int, 2> CreateTransitiveDependencies(xt::xtensor<int, 2> weights)
    {
        boost::adjacency_list<> dependencyGraph;
        for (const auto [u, v] : xt::argwhere(equal(weights, -1)))
            add_edge(v, u, dependencyGraph);

        boost::adjacency_list<> transitiveClosure;
        transitive_closure(dependencyGraph, transitiveClosure);

        for (const auto& e : make_iterator_range(edges(transitiveClosure)))
            weights(e.m_target, e.m_source) = -1;

        return weights;
    }
}
