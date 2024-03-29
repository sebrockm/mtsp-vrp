#include "DependencyHelpers.hpp"

#include "MtspModel.hpp"
#include "TsplpExceptions.hpp"

#include <HasCycle.hpp>

#include <boost/graph/transitive_closure.hpp>

namespace tsplp
{
xt::xtensor<double, 2> CreateTransitiveDependencies(xt::xtensor<double, 2> weights)
{
    boost::adjacency_list<> dependencyGraph;
    for (const auto [v, u] : xt::argwhere(equal(weights, -1)))
        add_edge(u, v, dependencyGraph);

    if (graph_algos::HasCycle(dependencyGraph))
        throw tsplp::CyclicDependenciesException {};

    boost::adjacency_list<> transitiveClosure;
    transitive_closure(dependencyGraph, transitiveClosure);

    for (const auto& e : make_iterator_range(edges(transitiveClosure)))
        weights(e.m_target, e.m_source) = -1;

    return weights;
}

DependencyGraph::DependencyGraph(const xt::xtensor<double, 2>& weights)
    : m_weights(weights)
{
    const auto N = weights.shape(0);

    m_node2incomingSpanMap.reserve(N);
    m_node2outgoingSpanMap.reserve(N);

    for (size_t u = 0; u < N; ++u)
    {
        const auto rangeBegin = ssize(m_outgoing);

        for (size_t v = 0; v < N; ++v)
        {
            if (weights(v, u) == -1)
                m_outgoing.push_back(v);
        }

        const auto rangeEnd = ssize(m_outgoing);
        m_node2outgoingSpanMap.emplace_back(rangeBegin, rangeEnd);
    }

    for (size_t u = 0; u < N; ++u)
    {
        const auto rangeBegin = ssize(m_incoming);

        for (size_t v = 0; v < N; ++v)
        {
            if (weights(u, v) == -1)
            {
                m_incoming.push_back(v);
                m_arcs.emplace_back(v, u);
            }
        }

        const auto rangeEnd = ssize(m_incoming);
        m_node2incomingSpanMap.emplace_back(rangeBegin, rangeEnd);
    }
}

std::span<const size_t> DependencyGraph::GetIncomingSpan(size_t n) const
{
    const auto [s, t] = m_node2incomingSpanMap[n];
    // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    return { m_incoming.data() + s, m_incoming.data() + t };
}

std::span<const size_t> DependencyGraph::GetOutgoingSpan(size_t n) const
{
    const auto [s, t] = m_node2outgoingSpanMap[n];
    // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    return { m_outgoing.data() + s, m_outgoing.data() + t };
}
}
