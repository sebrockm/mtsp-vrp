#pragma once

#include <xtensor/xtensor.hpp>

#include <span>

namespace tsplp
{
xt::xtensor<double, 2> CreateTransitiveDependencies(xt::xtensor<double, 2> weights);

class DependencyGraph
{
private:
    std::vector<std::pair<size_t, size_t>> m_arcs;
    std::vector<size_t> m_incoming;
    std::vector<size_t> m_outgoing;

    std::vector<std::pair<size_t, size_t>> m_node2incomingSpanMap;
    std::vector<std::pair<size_t, size_t>> m_node2outgoingSpanMap;

    const xt::xtensor<double, 2>& m_weights;

public:
    explicit DependencyGraph(const xt::xtensor<double, 2>& weights);

    [[nodiscard]] const auto& GetArcs() const { return m_arcs; }
    [[nodiscard]] std::span<const size_t> GetIncomingSpan(size_t n) const;
    [[nodiscard]] std::span<const size_t> GetOutgoingSpan(size_t n) const;

    [[nodiscard]] bool HasArc(size_t u, size_t v) const { return m_weights(v, u) == -1; }
};
}
