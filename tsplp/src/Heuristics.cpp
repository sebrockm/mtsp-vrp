#include "Heuristics.hpp"

#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/connected_components.hpp>
#include <boost/graph/topological_sort.hpp>
#include <boost/range/adaptor/reversed.hpp>

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
        add_edge(u, v, dependencyGraph);

    for (size_t a = 0; a < A; ++a)
        add_edge(static_cast<size_t>(startPositions[a]), static_cast<size_t>(endPositions[a]), dependencyGraph);

    assert(num_edges(dependencyGraph) >= size(dependencies));

    std::vector<size_t> componentIds(N);
    const auto numberOfComponents = boost::connected_components(dependencyGraph, componentIds.data());

    std::vector<size_t> order;
    boost::topological_sort(dependencyGraph, std::back_inserter(order));

    std::unordered_set<size_t> inserted;
    std::vector<size_t> component2AgentMap(static_cast<size_t>(numberOfComponents), A);

    auto paths = std::vector<std::vector<int>>(A);
    int cost = 0;
    for (size_t a = 0; a < A; ++a)
    {
        assert(componentIds[startPositions[a]] == componentIds[endPositions[a]]);

        paths[a].push_back(startPositions[a]);
        inserted.insert(static_cast<size_t>(startPositions[a]));

        paths[a].push_back(endPositions[a]);
        inserted.insert(static_cast<size_t>(endPositions[a]));

        cost += weights(startPositions[a], endPositions[a]);

        component2AgentMap[componentIds[static_cast<size_t>(startPositions[a])]] = a;
    }

    std::vector<size_t> lastInsertPositionOfComponent(static_cast<size_t>(numberOfComponents), 0);

    for (const auto n : boost::adaptors::reverse(order))
    {
        if (inserted.contains(n))
            continue;

        const auto comp = componentIds[n];

        auto minDeltaCost = std::numeric_limits<int>::max();
        auto minA = std::numeric_limits<size_t>::max();
        auto minI = std::numeric_limits<size_t>::max();

        const auto aRange = component2AgentMap[comp] == A
            ? std::make_pair(static_cast<size_t>(0), A)
            : std::make_pair(component2AgentMap[comp], component2AgentMap[comp] + 1);

        for (size_t a = aRange.first; a < aRange.second; ++a)
        {
            for (size_t i = 1 + lastInsertPositionOfComponent[comp]; i < paths[a].size(); ++i)
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

        component2AgentMap[comp] = minA;
        lastInsertPositionOfComponent[comp] = minI;
    }

    return { paths, cost };
}
