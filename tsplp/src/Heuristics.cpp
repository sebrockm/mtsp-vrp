#include "Heuristics.hpp"

#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/topological_sort.hpp>
#include <boost/graph/properties.hpp>

#include <xtensor/xview.hpp>

std::tuple<std::vector<std::vector<int>>, int> tsplp::NearestInsertion(
    const xt::xtensor<int, 2>& weights, const xt::xtensor<int, 1>& startPositions, const xt::xtensor<int, 1>& endPositions)
{
    const auto A = startPositions.size();
    assert(endPositions.size() == A);
    const auto N = weights.shape(0);
    assert(weights.shape(1) == N);

    boost::adjacency_list<boost::vecS, boost::vecS, boost::directedS, size_t> dependencyGraph;
    for (const auto [v, u] : xt::argwhere(equal(weights, -1)))
    {
        const auto uu = add_vertex(u, dependencyGraph);
        const auto vv = add_vertex(v, dependencyGraph);
        add_edge(uu, vv, dependencyGraph);
    }

    std::vector<size_t> order;
    boost::topological_sort(dependencyGraph, std::back_inserter(order));
    for (auto& i : order)
        i = dependencyGraph[i];

    auto paths = std::vector<std::vector<int>>(A);
    int cost = 0;
    for (size_t a = 0; a < A; ++a)
    {
        paths[a].reserve(N);
        if (a == 0)
            paths[a].insert(paths[a].end(), order.rbegin(), order.rend());

        for (size_t i = 0; i + 1 < paths[a].size(); ++i)
            cost += weights(paths[a][i], paths[a][i + 1]);
    }

    for (size_t n = 0; n < N; ++n)
    {
        if (std::find(order.begin(), order.end(), n) != order.end())
            continue;

        auto minDeltaCost = std::numeric_limits<int>::max();
        auto minA = std::numeric_limits<size_t>::max();
        auto minI = std::numeric_limits<size_t>::max();

        for (size_t a = 0; a < A; ++a)
        {
            for (size_t i = 1; i < paths[a].size(); ++i)
            {
                const auto oldCost = weights(paths[a][i - 1], paths[a][i]);
                if (oldCost < 0)
                    throw 0;
                const auto newCost1 = weights(paths[a][i - 1], n);
                if (newCost1 < 0)
                    throw 1;
                const auto newCost2 = weights(n, paths[a][i]);
                if (newCost2 < 0)
                    throw 2;
                const auto newCost = newCost1 + newCost2;
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
