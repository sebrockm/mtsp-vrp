#include "WeightManager.hpp"
#include "DependencyHelpers.hpp"

#include <unordered_set>

#include <xtensor/xindex_view.hpp>
#include <xtensor/xview.hpp>

tsplp::WeightManager::WeightManager(xt::xtensor<int, 2> weights, xt::xtensor<size_t, 1> originalStartPositions, xt::xtensor<size_t, 1> originalEndPositions)
    : m_weights(std::move(weights)), m_startPositions(originalStartPositions), m_endPositions(originalEndPositions), m_hasDependencies(xt::any(equal(m_weights, -1)))
{
    if (m_startPositions.size() != m_endPositions.size())
        throw std::runtime_error("Start and end positions must have the same size.");

    if (m_weights.shape(0) != m_weights.shape(1))
        throw std::runtime_error("The weights must have shape (N, N).");

    // ignore self referring arcs for convenience
    for (size_t n = 0; n < m_weights.shape(0); ++n)
        m_weights(n, n) = 0;

    const auto A = m_startPositions.size();

    std::unordered_set<int> startEndInUse;

    for (size_t a = 0; a < A; ++a)
    {
        if (const auto s = originalStartPositions[a]; startEndInUse.contains(s))
        {
            m_weights = xt::concatenate(xtuple(m_weights, xt::view(m_weights, s, xt::newaxis(), xt::all())), 0);
            m_weights = xt::concatenate(xtuple(m_weights, xt::view(m_weights, xt::all(), xt::newaxis(), s)), 1);
            m_startPositions[a] = m_weights.shape(0) - 1;
            m_toOriginal[m_startPositions[a]] = s;

            auto addedRow = xt::view(m_weights, -1, xt::all());
            xt::filtration(addedRow, equal(addedRow, -1)) = 0; // a start node copied from an end node must not have dependees
        }
        else
        {
            startEndInUse.insert(s);
        }

        if (const auto e = originalEndPositions[a]; startEndInUse.contains(e))
        {
            m_weights = xt::concatenate(xtuple(m_weights, xt::view(m_weights, e, xt::newaxis(), xt::all())), 0);
            m_weights = xt::concatenate(xtuple(m_weights, xt::view(m_weights, xt::all(), xt::newaxis(), e)), 1);
            m_endPositions[a] = m_weights.shape(0) - 1;
            m_toOriginal[m_endPositions[a]] = e;

            auto addedColumn = xt::view(m_weights, xt::all(), -1);
            xt::filtration(addedColumn, equal(addedColumn, -1)) = 0; // an end node copied from a start node must not have dependers
        }
        else
        {
            startEndInUse.insert(e);
        }
    }

    for (size_t a = 0; a < A; ++a)
    {
        m_weights(m_endPositions[a], m_startPositions[(a + 1) % A]) = 0; // artificially connect end positions i to start positions i+1 with zero cost

        // The following step looks beneficial, however, it is problematic, in particular in the case A == 1.
        // That is because the arc (e, s) must be used here to complete a full cycle and its weight must be 0.
        // Hence, we cannot use it to store the dependency flag (-1) of s->e.
        // But also in the case A > 1 this complicates the heuristics.
        // Fortunately, doing this is optional because the initial constraints already enforce a path from s to e.
        // So, we just leave it out.
        // TODO: what if initial weights already have a -1 here?
        // m_weights(m_endPositions[a], m_startPositions[a]) = -1; // set dependency s->e
    }

    m_weights = CreateTransitiveDependencies(std::move(m_weights));
}

std::vector<std::vector<size_t>> tsplp::WeightManager::TransformPathsBack(std::vector<std::vector<size_t>> paths) const
{
    for (auto& path : paths)
        for (auto& i : path)
            if (m_toOriginal.contains(i))
                i = m_toOriginal.at(i);

    return paths;
}
