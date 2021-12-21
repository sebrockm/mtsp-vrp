#include "DependencyHelpers.hpp"

#include <boost/graph/transitive_closure.hpp>

namespace
{
    template <bool reversed>
    auto CreateTransitiveClosureGraph(std::span<const std::pair<int, int>> dependencies)
    {
        boost::adjacency_list<> dependencyGraph;
        for (const auto [u, v] : dependencies)
        {
            if constexpr (reversed)
                add_edge(static_cast<size_t>(v), static_cast<size_t>(u), dependencyGraph);
            else
                add_edge(static_cast<size_t>(u), static_cast<size_t>(v), dependencyGraph);
        }

        boost::adjacency_list<> transitiveClosure;
        boost::transitive_closure(dependencyGraph, transitiveClosure);

        std::unordered_map<int, std::vector<int>> result;

        for (const auto v : boost::make_iterator_range(vertices(transitiveClosure)))
        {
            const auto [first, last] = adjacent_vertices(v, transitiveClosure);
            if (first != last)
            {
                const auto v2 = static_cast<int>(v);
                result.emplace(std::piecewise_construct, std::make_tuple(v2), std::make_tuple(first, last));
                std::sort(result[v2].begin(), result[v2].end());
            }
        }

        return result;
    }
}

namespace tsplp::graph
{
    std::unordered_map<int, std::vector<int>> CreateTransitiveDependencies(std::span<const std::pair<int, int>> dependencies)
    {
        return CreateTransitiveClosureGraph<false>(dependencies);
    }

    std::unordered_map<int, std::vector<int>> CreateTransitiveDependenciesReversed(std::span<const std::pair<int, int>> dependencies)
    {
        return CreateTransitiveClosureGraph<true>(dependencies);
    }
}