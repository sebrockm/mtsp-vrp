#pragma once

#include <span>
#include <unordered_map>
#include <vector>

namespace tsplp::graph
{
    std::unordered_map<int, std::vector<int>> CreateTransitiveDependencies(std::span<const std::pair<int, int>> dependencies);
}