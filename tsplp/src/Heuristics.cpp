#include "Heuristics.hpp"

#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/connected_components.hpp>
#include <boost/graph/topological_sort.hpp>
#include <boost/range/adaptor/reversed.hpp>

#include <unordered_set>

std::tuple<std::vector<std::vector<size_t>>, int> tsplp::NearestInsertion(
    const xt::xtensor<int, 2>& weights, const xt::xtensor<size_t, 1>& startPositions, const xt::xtensor<size_t, 1>& endPositions)
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
        add_edge(startPositions[a], endPositions[a], dependencyGraph);

    assert(num_edges(dependencyGraph) >= size(dependencies));

    std::vector<size_t> componentIds(N);
    const auto numberOfComponents = boost::connected_components(dependencyGraph, componentIds.data());

    std::vector<size_t> order;
    boost::topological_sort(dependencyGraph, std::back_inserter(order));

    std::vector<size_t> component2AgentMap(numberOfComponents, A);

    auto paths = std::vector<std::vector<size_t>>(A);
    int cost = 0;
    for (size_t a = 0; a < A; ++a)
    {
        assert(componentIds[startPositions[a]] == componentIds[endPositions[a]]);

        paths[a].push_back(startPositions[a]);
        paths[a].push_back(endPositions[a]);

        cost += weights(startPositions[a], endPositions[a]);
        component2AgentMap[componentIds[startPositions[a]]] = a;
    }

    std::vector<size_t> lastInsertPositionOfComponent(numberOfComponents, 0);

    for (const auto n : boost::adaptors::reverse(order))
    {
        if (std::find(startPositions.begin(), startPositions.end(), n) != startPositions.end()
            || std::find(endPositions.begin(), endPositions.end(), n) != endPositions.end())
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
        paths[minA].insert(paths[minA].begin() + static_cast<DiffT>(minI), n);
        cost += minDeltaCost;

        component2AgentMap[comp] = minA;
        lastInsertPositionOfComponent[comp] = minI;
    }

    return { paths, cost };
}
