#include "DependencyHelpers.hpp"

#include <boost/graph/transitive_closure.hpp>

namespace tsplp::graph
{
    std::unordered_map<int, std::vector<int>> CreateTransitiveDependencies(std::span<const std::pair<int, int>> dependencies)
    {
        boost::adjacency_list<> dependencyGraph;
        for (const auto [u, v] : dependencies)
            add_edge(static_cast<size_t>(u), static_cast<size_t>(v), dependencyGraph);

        boost::adjacency_list<> transitiveClosure;
        boost::transitive_closure(dependencyGraph, transitiveClosure);

        std::unordered_map<int, std::vector<int>> result;

        for (const auto v : boost::make_iterator_range(vertices(transitiveClosure)))
        {
            const auto [first, last] = adjacent_vertices(v, transitiveClosure);
            if (first != last)
                result.emplace(std::piecewise_construct, std::make_tuple(static_cast<int>(v)), std::make_tuple(first, last));
        }

        return result;
    }
}