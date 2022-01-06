#include "DependencyHelpers.hpp"
#include "MtspModel.hpp"

#include <boost/graph/depth_first_search.hpp>
#include <boost/graph/transitive_closure.hpp>

namespace
{
    template <typename Graph>
    void ThrowOnCycle(const Graph& graph)
    {
        struct CycleDetector : public boost::dfs_visitor<>
        {
            void back_edge(auto, auto)
            {
                throw tsplp::CyclicDependenciesException{};
            }
        };

        boost::depth_first_search(graph, visitor(CycleDetector{}));
    }
}

namespace tsplp
{
    xt::xtensor<int, 2> CreateTransitiveDependencies(xt::xtensor<int, 2> weights)
    {
        boost::adjacency_list<> dependencyGraph;
        for (const auto [u, v] : xt::argwhere(equal(weights, -1)))
            add_edge(v, u, dependencyGraph);

        ThrowOnCycle(dependencyGraph);

        boost::adjacency_list<> transitiveClosure;
        transitive_closure(dependencyGraph, transitiveClosure);

        for (const auto& e : make_iterator_range(edges(transitiveClosure)))
            weights(e.m_target, e.m_source) = -1;

        return weights;
    }
}
