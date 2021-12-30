#include "Heuristics.hpp"

#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/connected_components.hpp>
#include <boost/graph/topological_sort.hpp>

#include <unordered_set>

#include <xtensor/xview.hpp>

std::tuple<std::vector<std::vector<int>>, int> tsplp::NearestInsertion(
    const xt::xtensor<int, 2>& weights, const xt::xtensor<int, 1>& startPositions, const xt::xtensor<int, 1>& endPositions)
{
    const auto A = startPositions.size();
    assert(endPositions.size() == A);
    const auto N = weights.shape(0);
    assert(weights.shape(1) == N);

    boost::adjacency_list<boost::vecS, boost::vecS, boost::directedS> dependencyGraph(N);
    const auto dependencies = xt::argwhere(equal(weights, -1));
    for (const auto [v, u] : dependencies)
    {
        add_edge(u, v, dependencyGraph);
    }

    assert(num_edges(dependencyGraph) == size(dependencies));

    std::vector<int> componentIds(num_vertices(dependencyGraph));
    [[maybe_unused]] const auto numberOfComponents = boost::connected_components(dependencyGraph, componentIds.data());

    std::vector<size_t> order;
    boost::topological_sort(dependencyGraph, std::back_inserter(order));

    std::unordered_set<int> inserted;

    auto paths = std::vector<std::vector<int>>(A);
    int cost = 0;
    for (size_t a = 0; a < A; ++a)
    {
        auto newEnd = std::remove(order.begin(), order.end(), startPositions[a]);
        newEnd = std::remove(order.begin(), newEnd, endPositions[a]);

        paths[a].reserve(N);
        paths[a].push_back(startPositions[a]);
        if (a == 0)
            paths[a].insert(paths[a].end(), std::make_reverse_iterator(newEnd), order.rend());  // TODO: better distribute the connected components among the agents
                                                                                                // also: handle cases when stard/end nodes have dependencies
        paths[a].push_back(endPositions[a]);

        inserted.insert(paths[a].begin(), paths[a].end());

        for (size_t i = 0; i + 1 < paths[a].size(); ++i)
            cost += weights(paths[a][i], paths[a][i + 1]);
    }

    for (size_t n = 0; n < N; ++n)
    {
        if (inserted.contains(static_cast<int>(n)))
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
