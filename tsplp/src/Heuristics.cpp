#include "Heuristics.hpp"

#include "DependencyHelpers.hpp"
#include "TsplpExceptions.hpp"

#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/connected_components.hpp>
#include <boost/graph/topological_sort.hpp>
#include <boost/range/adaptor/reversed.hpp>
#include <xtensor/xmanipulation.hpp>
#include <xtensor/xview.hpp>

#include <unordered_set>

namespace tsplp
{

std::vector<std::vector<size_t>> ExploitFractionalSolution(
    OptimizationMode optimizationMode, xt::xarray<double> fractionalSolution,
    xt::xarray<double> weights, const xt::xtensor<size_t, 1>& startPositions,
    const xt::xtensor<size_t, 1>& endPositions, const DependencyGraph& dependencies,
    std::chrono::steady_clock::time_point endTime)
{
    const auto A = startPositions.size();
    assert(endPositions.size() == A);

    if (weights.dimension() == 2)
        weights = xt::repeat(xt::view(weights, xt::newaxis(), xt::all()), A, 0);

    const auto [heuristicPaths, _] = NearestInsertion(
        optimizationMode, (1.0 - fractionalSolution) * weights, startPositions, endPositions,
        dependencies, endTime);

    return heuristicPaths;
}

std::tuple<std::vector<std::vector<size_t>>, double> NearestInsertionSum(
    xt::xarray<double> weights, const xt::xtensor<size_t, 1>& startPositions,
    const xt::xtensor<size_t, 1>& endPositions, const DependencyGraph& dependencies,
    std::chrono::steady_clock::time_point endTime)
{
    const auto A = startPositions.size();
    assert(endPositions.size() == A);

    if (weights.dimension() == 2)
        weights = xt::repeat(xt::view(weights, xt::newaxis(), xt::all()), A, 0);

    assert(weights.dimension() == 3);

    const auto N = weights.shape(1);
    assert(weights.shape(2) == N);

    boost::adjacency_list<boost::vecS, boost::vecS, boost::directedS> dependencyGraph(N);
    boost::adjacency_list<boost::vecS, boost::vecS, boost::undirectedS> dependencyGraphUndirected(
        N);
    for (const auto& [u, v] : dependencies.GetArcs())
    {
        add_edge(u, v, dependencyGraph);
        add_edge(u, v, dependencyGraphUndirected);
    }

    for (size_t a = 0; a < A; ++a)
    {
        add_edge(startPositions[a], endPositions[a], dependencyGraph);
        add_edge(startPositions[a], endPositions[a], dependencyGraphUndirected);
    }

    assert(num_edges(dependencyGraph) >= size(dependencies.GetArcs()));

    std::vector<size_t> componentIds(N);
    const auto numberOfComponents
        = boost::connected_components(dependencyGraphUndirected, componentIds.data());

    std::vector<size_t> order;
    boost::topological_sort(dependencyGraph, std::back_inserter(order));

    std::vector<size_t> component2AgentMap(numberOfComponents, A);

    auto paths = std::vector<std::vector<size_t>>(A);
    double cost = 0;
    for (size_t a = 0; a < A; ++a)
    {
        assert(componentIds[startPositions[a]] == componentIds[endPositions[a]]);

        paths[a].push_back(startPositions[a]);
        paths[a].push_back(endPositions[a]);

        cost += weights(a, startPositions[a], endPositions[a]);

        if (component2AgentMap[componentIds[startPositions[a]]] != A)
            throw IncompatibleDependenciesException();

        component2AgentMap[componentIds[startPositions[a]]] = a;
    }

    std::vector<size_t> lastInsertPositionOfComponent(numberOfComponents, 0);

    for (const auto n : boost::adaptors::reverse(order))
    {
        if (std::chrono::steady_clock::now() >= endTime)
            return { std::vector<std::vector<size_t>> {}, 0 };

        if (std::find(startPositions.begin(), startPositions.end(), n) != startPositions.end()
            || std::find(endPositions.begin(), endPositions.end(), n) != endPositions.end())
            continue;

        const auto comp = componentIds[n];

        auto minDeltaCost = std::numeric_limits<double>::max();
        auto minA = std::numeric_limits<size_t>::max();
        auto minI = std::numeric_limits<size_t>::max();

        const auto [aRangeFirst, aRangeLast] = component2AgentMap[comp] == A
            ? std::make_pair(static_cast<size_t>(0), A)
            : std::make_pair(component2AgentMap[comp], component2AgentMap[comp] + 1);

        for (size_t a = aRangeFirst; a < aRangeLast; ++a)
        {
            for (size_t i = 1 + lastInsertPositionOfComponent[comp]; i < paths[a].size(); ++i)
            {
                const auto oldCost = weights(a, paths[a][i - 1], paths[a][i]);
                const auto newCost = weights(a, paths[a][i - 1], n) + weights(a, n, paths[a][i]);
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

std::tuple<std::vector<std::vector<size_t>>, double> NearestInsertionMax(
    xt::xarray<double> weights, const xt::xtensor<size_t, 1>& startPositions,
    const xt::xtensor<size_t, 1>& endPositions, const DependencyGraph& dependencies,
    std::chrono::steady_clock::time_point endTime)
{
    const auto A = startPositions.size();
    assert(endPositions.size() == A);

    if (weights.dimension() == 2)
        weights = xt::repeat(xt::view(weights, xt::newaxis(), xt::all()), A, 0);

    assert(weights.dimension() == 3);

    const auto N = weights.shape(1);
    assert(weights.shape(2) == N);

    boost::adjacency_list<boost::vecS, boost::vecS, boost::directedS> dependencyGraph(N);
    boost::adjacency_list<boost::vecS, boost::vecS, boost::undirectedS> dependencyGraphUndirected(
        N);
    for (const auto& [u, v] : dependencies.GetArcs())
    {
        add_edge(u, v, dependencyGraph);
        add_edge(u, v, dependencyGraphUndirected);
    }

    for (size_t a = 0; a < A; ++a)
    {
        add_edge(startPositions[a], endPositions[a], dependencyGraph);
        add_edge(startPositions[a], endPositions[a], dependencyGraphUndirected);
    }

    assert(num_edges(dependencyGraph) >= size(dependencies.GetArcs()));

    std::vector<size_t> componentIds(N);
    const auto numberOfComponents
        = boost::connected_components(dependencyGraphUndirected, componentIds.data());

    std::vector<size_t> order;
    boost::topological_sort(dependencyGraph, std::back_inserter(order));

    std::vector<size_t> component2AgentMap(numberOfComponents, A);

    auto paths = std::vector<std::vector<size_t>>(A);
    auto pathLengths = std::vector<double>(A);
    size_t longestA = A;
    double cost = 0;
    for (size_t a = 0; a < A; ++a)
    {
        assert(componentIds[startPositions[a]] == componentIds[endPositions[a]]);

        paths[a].push_back(startPositions[a]);
        paths[a].push_back(endPositions[a]);

        pathLengths[a] = weights(a, startPositions[a], endPositions[a]);
        if (pathLengths[a] > cost)
        {
            longestA = a;
            cost = pathLengths[a];
        }

        if (component2AgentMap[componentIds[startPositions[a]]] != A)
            throw IncompatibleDependenciesException();

        component2AgentMap[componentIds[startPositions[a]]] = a;
    }

    std::vector<size_t> lastInsertPositionOfComponent(numberOfComponents, 0);

    for (const auto n : boost::adaptors::reverse(order))
    {
        if (std::chrono::steady_clock::now() >= endTime)
            return { std::vector<std::vector<size_t>> {}, 0 };

        if (std::find(startPositions.begin(), startPositions.end(), n) != startPositions.end()
            || std::find(endPositions.begin(), endPositions.end(), n) != endPositions.end())
            continue;

        const auto comp = componentIds[n];

        auto minDeltaCost = std::numeric_limits<double>::max();
        auto minA = std::numeric_limits<size_t>::max();
        auto minI = std::numeric_limits<size_t>::max();

        const auto [aRangeFirst, aRangeLast] = component2AgentMap[comp] == A
            ? std::make_pair(static_cast<size_t>(0), A)
            : std::make_pair(component2AgentMap[comp], component2AgentMap[comp] + 1);

        for (size_t a = aRangeFirst; a < aRangeLast; ++a)
        {
            for (size_t i = 1 + lastInsertPositionOfComponent[comp]; i < paths[a].size(); ++i)
            {
                const auto oldCost = weights(a, paths[a][i - 1], paths[a][i]);
                const auto newCost = weights(a, paths[a][i - 1], n) + weights(a, n, paths[a][i]);
                const auto deltaCost
                    = std::max(newCost - oldCost + pathLengths[a] - pathLengths[longestA], 0.0);

                if (deltaCost < minDeltaCost)
                {
                    minDeltaCost = deltaCost;
                    minA = a;
                    minI = i;
                    if (minDeltaCost == 0)
                        break;
                }
            }

            if (minDeltaCost == 0)
                break;
        }

        using DiffT = decltype(paths[minA].begin())::difference_type;
        paths[minA].insert(paths[minA].begin() + static_cast<DiffT>(minI), n);
        cost += minDeltaCost;
        if (minDeltaCost > 0)
            longestA = minA;

        component2AgentMap[comp] = minA;
        lastInsertPositionOfComponent[comp] = minI;
    }

    return { paths, cost };
}

std::tuple<std::vector<std::vector<size_t>>, double> NearestInsertion(
    OptimizationMode optimizationMode, xt::xarray<double> weights,
    const xt::xtensor<size_t, 1>& startPositions, const xt::xtensor<size_t, 1>& endPositions,
    const DependencyGraph& dependencies, std::chrono::steady_clock::time_point endTime)
{
    switch (optimizationMode)
    {
    case OptimizationMode::Max:
        return NearestInsertionMax(weights, startPositions, endPositions, dependencies, endTime);
    case OptimizationMode::Sum:
        return NearestInsertionSum(weights, startPositions, endPositions, dependencies, endTime);
    default:
        throw TsplpException();
    }
}

class PathsInfo
{
    OptimizationMode m_optimizationMode;
    size_t m_longestA = 0;
    std::vector<double> m_pathLengths;

public:
    PathsInfo(
        OptimizationMode optimizationMode, const std::vector<std::vector<size_t>>& paths,
        const xt::xarray<double> weights)
        : m_optimizationMode(optimizationMode)
    {
        const auto A = paths.size();
        m_pathLengths.resize(A);
        if (m_optimizationMode == OptimizationMode::Max)
        {
            for (size_t a = 0; a < A; ++a)
            {
                m_pathLengths[a] = CalculatePathLength(paths[a], weights);

                if (m_pathLengths[a] > m_pathLengths[m_longestA])
                    m_longestA = a;
            }
        }
    }

    [[nodiscard]] std::array<size_t, 2> GetA2Range(size_t a1) const
    {
        switch (m_optimizationMode)
        {
        case OptimizationMode::Sum:
            return { a1, m_pathLengths.size() };
        case OptimizationMode::Max:
            if (a1 < m_longestA)
                return { m_longestA, m_longestA + 1 };
            if (a1 == m_longestA)
                return { m_longestA, m_pathLengths.size() };
            if (a1 > m_longestA)
                return { 0, 0 };
        }

        return { 0, 0 };
    }

    [[nodiscard]] double CalculateOverallImprovement(
        size_t a1, double improvementA1, size_t a2, double improvementA2) const
    {
        if (m_optimizationMode == OptimizationMode::Sum)
            return improvementA1 + improvementA2;

        const auto oldObjective = m_pathLengths[m_longestA];
        double newObjective = 0.0;
        for (size_t a = 0; a < m_pathLengths.size(); ++a)
        {
            auto aLength = m_pathLengths[a];
            if (a == a1)
                aLength -= improvementA1;
            if (a == a2)
                aLength -= improvementA2;

            if (aLength > newObjective)
                newObjective = aLength;
        }

        return oldObjective - newObjective;
    }

    void ApplyImprovement(size_t a1, double improvementA1, size_t a2, double improvementA2)
    {
        if (m_optimizationMode == OptimizationMode::Sum)
            return;

        m_pathLengths[a1] -= improvementA1;
        m_pathLengths[a2] -= improvementA2;
        m_longestA = 0;
        for (size_t a = 1; a < m_pathLengths.size(); ++a)
        {
            if (m_pathLengths[a] > m_pathLengths[m_longestA])
                m_longestA = a;
        }
    }
};

std::tuple<std::vector<std::vector<size_t>>, double> TwoOptPaths(
    OptimizationMode optimizationMode, std::vector<std::vector<size_t>> paths,
    xt::xarray<double> weights, const DependencyGraph& dependencies,
    std::chrono::steady_clock::time_point endTime)
{
    assert(weights.dimension() == 2);

    const auto A = paths.size();
    assert(A > 0);

    PathsInfo pathsInfo(optimizationMode, paths, weights);

    bool hasImproved = true;
    double improvementSum = 0.0;

    while (hasImproved && std::chrono::steady_clock::now() < endTime)
    {
        hasImproved = false;

        for (size_t a1 = 0; a1 < A; ++a1)
        {
            const auto [a2Start, a2End] = pathsInfo.GetA2Range(a1);
            for (size_t a2 = a2Start; a2 < a2End; ++a2)
            {
                for (size_t i = 1; i < paths[a1].size() - 1; ++i)
                {
                    const auto u = paths[a1][i];
                    if (a1 != a2
                        && (!dependencies.GetIncomingSpan(u).empty()
                            || !dependencies.GetOutgoingSpan(u).empty()))
                        continue;

                    const auto jStart = a1 == a2 ? i + 1 : static_cast<size_t>(1);
                    // TODO: Consider involving start and end nodes here unless enforced by
                    // dependencies
                    for (size_t j = jStart; j < paths[a2].size() - 1; ++j)
                    {
                        const auto v = paths[a2][j];

                        if (a1 != a2
                            && (!dependencies.GetIncomingSpan(v).empty()
                                || !dependencies.GetOutgoingSpan(v).empty()))
                            continue;

                        if (a1 == a2)
                        {
                            bool wouldBreakDependency = false;
                            for (auto k = i; k < j; ++k)
                            {
                                if (dependencies.HasArc(u, paths[a1][k + 1])
                                    || dependencies.HasArc(paths[a1][k], v))
                                {
                                    wouldBreakDependency = true;
                                    break;
                                }
                            }

                            if (wouldBreakDependency)
                                continue;
                        }

                        double improvementA1 = 0.0;
                        double improvementA2 = 0.0;

                        if (a1 == a2 && j == i + 1)
                        {
                            // clang-format off
                            const auto before1 = weights(paths[a1][i - 1], paths[a1][i    ]);
                            const auto before2 = weights(paths[a1][i    ], paths[a1][i + 1]);
                            const auto before4 = weights(paths[a1][i + 1], paths[a1][i + 2]);
                            const auto after1  = weights(paths[a1][i - 1], paths[a1][i + 1]);
                            const auto after2  = weights(paths[a1][i + 1], paths[a1][i    ]);
                            const auto after4  = weights(paths[a1][i    ], paths[a1][i + 2]);
                            improvementA1 = before1 + before2 + before4 - after1 - after2 - after4;
                            // clang-format on
                        }
                        else
                        {
                            // clang-format off
                            const auto before1 = weights(paths[a1][i - 1], paths[a1][i    ]);
                            const auto before2 = weights(paths[a1][i    ], paths[a1][i + 1]);
                            const auto before3 = weights(paths[a2][j - 1], paths[a2][j    ]);
                            const auto before4 = weights(paths[a2][j    ], paths[a2][j + 1]);
                            const auto after1  = weights(paths[a1][i - 1], paths[a2][j    ]);
                            const auto after2  = weights(paths[a2][j    ], paths[a1][i + 1]);
                            const auto after3  = weights(paths[a2][j - 1], paths[a1][i    ]);
                            const auto after4  = weights(paths[a1][i    ], paths[a2][j + 1]);
                            improvementA1 = before1 + before2 - after1 - after2;
                            improvementA2 = before3 + before4 - after3 - after4;
                            // clang-format on
                        }

                        assert(weights(paths[a1][i], paths[a1][i]) == 0);

                        const auto improvement = pathsInfo.CalculateOverallImprovement(
                            a1, improvementA1, a2, improvementA2);

                        if (improvement > 0)
                        {
                            hasImproved = true;
                            std::swap(paths[a1][i], paths[a2][j]);
                            pathsInfo.ApplyImprovement(a1, improvementA1, a2, improvementA2);
                            improvementSum += improvement;
                        }
                    }
                }
            }
        }
    }

    return { paths, improvementSum };
}

double CalculatePathLength(const std::vector<size_t>& path, const xt::xarray<double> weights)
{
    [[maybe_unused]] const auto N = weights.shape(0);
    assert(weights.dimension() == 2);
    assert(weights.shape(1) == N);

    double length = 0.0;
    for (size_t i = 1; i < path.size(); ++i)
    {
        assert(path[i - 1] < N);
        assert(path[i] < N);
        length += weights(path[i - 1], path[i]);
    }

    return length;
}

double CalculateObjective(
    const OptimizationMode optimizationMode, const std::vector<std::vector<size_t>>& paths,
    const xt::xarray<double> weights)
{
    const auto A = paths.size();

    assert(weights.dimension() == 2 || weights.dimension() == 3);
    assert(weights.dimension() == 2 || weights.shape(0) == A);

    double objective = 0.0;
    for (size_t a = 0; a < A; ++a)
    {
        const auto pathLength = CalculatePathLength(
            paths[a], weights.dimension() == 2 ? weights : xt::view(weights, a));

        switch (optimizationMode)
        {
        case OptimizationMode::Sum:
            objective += pathLength;
            break;
        case OptimizationMode::Max:
            objective = std::max(pathLength, objective);
            break;
        }
    }

    return objective;
}
}