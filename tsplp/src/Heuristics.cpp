#include "Heuristics.hpp"

std::tuple<std::vector<std::vector<int>>, int> tsplp::NearestInsertion(
    const xt::xtensor<int, 2>& weights, const xt::xtensor<int, 1>& startPositions, const xt::xtensor<int, 1>& endPositions)
{
    const auto A = startPositions.size();
    assert(endPositions.size() == A);
    const auto N = weights.shape(0);
    assert(weights.shape(1) == N);

    auto paths = std::vector<std::vector<int>>(A);
    int cost = 0;
    for (size_t a = 0; a < A; ++a)
    {
        paths[a].reserve(N);
        paths[a].push_back(startPositions[a]);
        paths[a].push_back(endPositions[a]);
        cost += weights(startPositions[a], endPositions[a]);
    }

    for (size_t n = 0; n < N; ++n)
    {
        if (std::find(startPositions.begin(), startPositions.end(), n) != startPositions.end()
            || std::find(endPositions.begin(), endPositions.end(), n) != endPositions.end())
            continue;

        auto minDeltaCost = std::numeric_limits<int>::max();
        auto minA = std::numeric_limits<size_t>::max();
        auto minI = std::numeric_limits<size_t>::max();

        for (size_t a = 0; a < A; ++a)
        {
            for (size_t i = 1; i < paths[a].size(); ++i)
            {
                const auto oldCost = weights(paths[a][i - 1], paths[a][i]);
                const auto newCost = weights(paths[a][i - 1], n) + weights(n, paths[a][i]);
                const auto deltaCost = newCost - oldCost;
                if (deltaCost < minDeltaCost)
                {
                    minDeltaCost = deltaCost;
                    minA = a;
                    minI = i;
                }
            }
        }

        using DiffT = decltype(paths[minA].begin())::difference_type;
        paths[minA].insert(paths[minA].begin() + static_cast<DiffT>(minI), static_cast<int>(n));
        cost += minDeltaCost;
    }

    return { paths, cost };
}
