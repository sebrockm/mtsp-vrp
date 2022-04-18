#include "DependencyHelpers.hpp"
#include "MtspModel.hpp"
#include "TsplpExceptions.hpp"

#include <boost/graph/depth_first_search.hpp>
#include <boost/graph/transitive_closure.hpp>

namespace
{
    struct CycleDetector : public boost::dfs_visitor<>
    {
        void back_edge(auto, auto)
        {
            throw tsplp::CyclicDependenciesException{};
        }
    };

    template <typename Graph>
    void ThrowOnCycle(const Graph& graph)
    {
        boost::depth_first_search(graph, visitor(CycleDetector{}));
    }
}

namespace tsplp
{
    xt::xtensor<int, 2> CreateTransitiveDependencies(xt::xtensor<int, 2> weights)
    {
        boost::adjacency_list<> dependencyGraph;
        for (const auto [v, u] : xt::argwhere(equal(weights, -1)))
            add_edge(u, v, dependencyGraph);

        ThrowOnCycle(dependencyGraph);

        boost::adjacency_list<> transitiveClosure;
        transitive_closure(dependencyGraph, transitiveClosure);

        for (const auto& e : make_iterator_range(edges(transitiveClosure)))
            weights(e.m_target, e.m_source) = -1;

        return weights;
    }

    DependencyGraph::DependencyGraph(const xt::xtensor<int, 2>& weights)
        : m_weights(weights)
    {
        const auto N = weights.shape(0);

        m_node2incomingSpanMap.reserve(N);
        m_node2outgoingSpanMap.reserve(N);

        for (size_t u = 0; u < N; ++u)
        {
            const auto rangeBegin = m_outgoing.size();

            for (size_t v = 0; v < N; ++v)
                if (weights(v, u) == -1)
                    m_outgoing.push_back(v);

            const auto rangeEnd = m_outgoing.size();
            m_node2outgoingSpanMap.emplace_back(m_outgoing.cbegin() + rangeBegin, m_outgoing.cbegin() + rangeEnd);
        }

        for (size_t u = 0; u < N; ++u)
        {
            const auto rangeBegin = m_incoming.size();

            for (size_t v = 0; v < N; ++v)
            {
                if (weights(u, v) == -1)
                {
                    m_incoming.push_back(v);
                    m_arcs.emplace_back(v, u);
                }
            }

            const auto rangeEnd = m_incoming.size();
            m_node2incomingSpanMap.emplace_back(m_incoming.cbegin() + rangeBegin, m_incoming.cbegin() + rangeEnd);
        }
    }
}
