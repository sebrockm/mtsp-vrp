#include "WeightManager.hpp"

#include <unordered_set>

#include <xtensor/xview.hpp>

tsplp::WeightManager::WeightManager(xt::xtensor<int, 2> weights, xt::xtensor<int, 1> startPositions, xt::xtensor<int, 1> endPositions)
    : m_weights(std::move(weights)), m_startPositions(startPositions), m_endPositions(endPositions)
{
    if (m_startPositions.size() != m_endPositions.size())
        throw std::runtime_error("Start and end positions must have the same size.");

    if (m_weights.shape(0) != m_weights.shape(1))
        throw std::runtime_error("The weights must have shape (N, N).");

    const auto A = m_startPositions.size();

    std::unordered_set<int> startEndInUse;

    for (size_t a = 0; a < A; ++a)
    {
        const auto s = startPositions[a];
        const auto e = endPositions[a];

        if (startEndInUse.contains(s))
        {
            m_weights = xt::concatenate(xtuple(m_weights, xt::view(m_weights, s, xt::newaxis(), xt::all())), 0);
            m_weights = xt::concatenate(xtuple(m_weights, xt::view(m_weights, xt::all(), xt::newaxis(), s)), 1);
            m_startPositions[a] = static_cast<int>(m_weights.shape(0)) - 1;
            m_toOriginal[m_startPositions[a]] = s;
        }
        else
        {
            startEndInUse.insert(s);
        }

        if (startEndInUse.contains(e))
        {
            m_weights = xt::concatenate(xtuple(m_weights, xt::view(m_weights, e, xt::newaxis(), xt::all())), 0);
            m_weights = xt::concatenate(xtuple(m_weights, xt::view(m_weights, xt::all(), xt::newaxis(), e)), 1);
            m_endPositions[a] = static_cast<int>(m_weights.shape(0)) - 1;
            m_toOriginal[m_endPositions[a]] = e;
        }
        else
        {
            startEndInUse.insert(e);
        }
    }

    // ignore self referring arcs for convenience
    for (size_t n = 0; n < m_weights.shape(0); ++n)
        m_weights(n, n) = 0;

    // artificially connect end positions i to start positions i+1 with zero cost
    for (size_t a = 0; a < A; ++a)
        m_weights(m_endPositions[a], m_startPositions[(a + 1) % A]) = 0;
}

std::vector<std::vector<int>> tsplp::WeightManager::TransformPathsBack(std::vector<std::vector<int>> paths) const
{
    for (auto& path : paths)
        for (auto& i : path)
            if (m_toOriginal.contains(i))
                i = m_toOriginal.at(i);

    return paths;
}
