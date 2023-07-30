#pragma once

#include <functional>
#include <span>

namespace graph_algos
{

// The input graph is a full undirected graph. It is passed as a number of nodes N and a range of
// doubles that stores the edge capacities as a lower triangular matrix, i.e. the capacity of an
// edge (u, v) with u < v is stored at index u * (u - 1) / 2 + v.
// The resulting Gomory Hu Tree is not returned directly. Instead, the passed callback is called
// whenever a new edge/cut has been generated. The parameters of the callback are the newly
// generated edge (u, v), the cut size aka weight of the edge, and the two cut components the the
// edge vertices u and v reside in.
void CreateGomoryHuTree(
    size_t N, std::span<const double> weights,
    const std::function<bool(
        size_t u, size_t v, double cutSize, std::span<const size_t> compU,
        std::span<const size_t> compV)>& newEdgeCallback);
}
