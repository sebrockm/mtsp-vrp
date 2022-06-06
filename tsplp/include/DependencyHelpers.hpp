#pragma once

#include <xtensor/xtensor.hpp>

#include <span>

namespace tsplp
{
xt::xtensor<int, 2> CreateTransitiveDependencies(xt::xtensor<int, 2> weights);

class DependencyGraph
{
private:
    std::vector<std::pair<size_t, size_t>> m_arcs;
    std::vector<size_t> m_incoming;
    std::vector<size_t> m_outgoing;

    std::vector<std::span<const size_t>> m_node2incomingSpanMap;
    std::vector<std::span<const size_t>> m_node2outgoingSpanMap;

    const xt::xtensor<int, 2>& m_weights;

public:
    explicit DependencyGraph(const xt::xtensor<int, 2>& weights);

    const auto& GetArcs() const { return m_arcs; }
    auto GetIncomingSpan(size_t n) const { return m_node2incomingSpanMap[n]; }
    auto GetOutgoingSpan(size_t n) const { return m_node2outgoingSpanMap[n]; }

    bool HasArc(size_t u, size_t v) const { return m_weights(v, u) == -1; }
};
}
