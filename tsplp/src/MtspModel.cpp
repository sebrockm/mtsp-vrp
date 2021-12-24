#include "MtspModel.hpp"
#include "Heuristics.hpp"
#include "LinearConstraint.hpp"
#include "SeparationAlgorithms.hpp"

#include <cassert>
#include <cmath>
#include <queue>
#include <stdexcept>
#include <xtensor/xadapt.hpp>
#include <xtensor/xview.hpp>

namespace
{
    void FixVariables(std::span<tsplp::Variable> variables, double value)
    {
        for (auto v : variables)
        {
            v.SetLowerBound(value);
            v.SetUpperBound(value);
        }
    }

    void UnfixVariables(std::span<tsplp::Variable> variables)
    {
        for (auto v : variables)
        {
            v.SetLowerBound(0.0);
            v.SetUpperBound(1.0);
        }
    }

    std::optional<tsplp::Variable> FindFractionalVariable(std::span<const tsplp::Variable> variables, double epsilon = 1.e-10)
    {
        std::optional<tsplp::Variable> closest = std::nullopt;
        double minAbs = 1.0;
        for (auto v : variables)
        {
            if (epsilon <= v.GetObjectiveValue() && v.GetObjectiveValue() <= 1.0 - epsilon)
            {
                if (std::abs(v.GetObjectiveValue() - 0.5) < minAbs)
                {
                    minAbs = std::abs(v.GetObjectiveValue() - 0.5);
                    closest = v;
                    if (minAbs < epsilon)
                        break;
                }
            }
        }

        return closest;
    }
}

tsplp::MtspModel::MtspModel(xt::xtensor<int, 1> startPositions, xt::xtensor<int, 1> endPositions, xt::xtensor<int, 2> weights)
    : m_weightsManager(std::move(weights), std::move(startPositions), std::move(endPositions)),
    A(m_weightsManager.A()),
    N(m_weightsManager.N()),
    m_model(A * N * N),
    X(xt::adapt(m_model.GetVariables(), { A, N, N })),
    m_objective(xt::sum(m_weightsManager.W() * X)())
{
    m_model.SetObjective(m_objective);

    std::vector<LinearConstraint> constraints;
    const auto numberOfConstraints = A * N + 2 * N + 3 * A + N * (N - 1) / 2;
    constraints.reserve(numberOfConstraints);

    // don't use self referring arcs (entries on diagonal)
    for (size_t a = 0; a < A; ++a)
        for (size_t n = 0; n < N; ++n)
            constraints.emplace_back(X(a, n, n) == 0);

    // degree inequalities
    for (size_t n = 0; n < N; ++n)
    {
        constraints.emplace_back(xt::sum(xt::view(X + 0, xt::all(), xt::all(), n))() == 1);
        constraints.emplace_back(xt::sum(xt::view(X + 0, xt::all(), n, xt::all()))() == 1);
    }

    // special inequalities for start and end nodes
    for (size_t a = 0; a < A; ++a)
    {
        // We write X + 0 instead of X to turn summed up type from Variable to LinearVariableComposition.
        // That is necessary because xtensor initializes the sum with a conversion from 0 to ResultType and we
        // don't provide a conversion from int to Variable, but we do provide one from int to LinearVariableCompositon.
        constraints.emplace_back(xt::sum(xt::view(X + 0, a, m_weightsManager.StartPositions()[a], xt::all()))() == 1); // arcs out of start nodes
        constraints.emplace_back(xt::sum(xt::view(X + 0, a, xt::all(), m_weightsManager.EndPositions()[a]))() == 1); // arcs into end nodes
        constraints.emplace_back(X(a, m_weightsManager.EndPositions()[a], m_weightsManager.StartPositions()[(a + 1) % A]) == 1); // artificial connections from end to next start
    }

    // inequalities to disallow cycles of length 2
    for (size_t u = 0; u < N; ++u)
        for (size_t v = u + 1; v < N; ++v)
            constraints.emplace_back((xt::sum(xt::view(X + 0, xt::all(), u, v)) + xt::sum(xt::view(X + 0, xt::all(), v, u)))() <= 1);

    m_model.AddConstraints(constraints);
}

tsplp::MtspResult tsplp::MtspModel::BranchAndCutSolve(std::chrono::milliseconds timeout, std::optional<int> heuristicObjective, std::optional<std::vector<std::vector<int>>> heuristicPaths)
{
    if (heuristicObjective.has_value() != heuristicPaths.has_value())
        throw std::runtime_error("If you provide a heuristic objective, you also have to provide a corresponding heuristic path.");

    if (heuristicPaths.has_value() && heuristicPaths->size() != A)
        throw std::runtime_error("Invalid heuristic paths");

    const auto startTime = std::chrono::steady_clock::now();

    MtspResult bestResult{};

    if (heuristicObjective.has_value())
    {
        bestResult.UpperBound = static_cast<double>(*heuristicObjective);
        bestResult.Paths = std::move(*heuristicPaths);
    }
    else
    {
        auto [nearestInsertionPaths, nearestInsertionObjective] = NearestInsertion(m_weightsManager.W(), m_weightsManager.StartPositions(), m_weightsManager.EndPositions());

        bestResult.Paths = std::move(nearestInsertionPaths);
        bestResult.UpperBound = static_cast<double>(nearestInsertionObjective);
    }

    if (std::chrono::steady_clock::now() >= startTime + timeout)
    {
        bestResult.IsTimeoutHit = true;
        return bestResult;
    }

    if (m_model.Solve() != Status::Optimal)
        return bestResult;

    bestResult.LowerBound = m_objective.Evaluate();
    std::vector<Variable> fixedVariables0{};
    std::vector<Variable> fixedVariables1{};

    struct SData
    {
        double lowerBound = -std::numeric_limits<double>::max();
        std::vector<Variable> fixedVariables0{};
        std::vector<Variable> fixedVariables1{};
        bool operator>(SData const& sd) const { return lowerBound > sd.lowerBound; }
    };

    std::priority_queue<SData, std::vector<SData>, std::greater<>> queue{};
    queue.emplace(bestResult.LowerBound, fixedVariables0, fixedVariables1);

    while (!queue.empty() && std::chrono::steady_clock::now() < startTime + timeout)
    {
        UnfixVariables(fixedVariables0);
        UnfixVariables(fixedVariables1);

        bestResult.LowerBound = queue.top().lowerBound;
        fixedVariables0 = queue.top().fixedVariables0;
        fixedVariables1 = queue.top().fixedVariables1;
        queue.pop();

        FixVariables(fixedVariables0, 0.0);
        FixVariables(fixedVariables1, 1.0);

        if (m_model.Solve() != Status::Optimal)
            continue;

        auto currentLowerBound = std::ceil(m_objective.Evaluate() - 1.e-10);

        // do exploiting here

        bestResult.LowerBound = std::min(bestResult.UpperBound, currentLowerBound);
        if (!queue.empty())
            bestResult.LowerBound = std::min(currentLowerBound, queue.top().lowerBound);

        if (bestResult.LowerBound >= bestResult.UpperBound)
            break;

        if (currentLowerBound >= bestResult.UpperBound)
            continue;

        // fix variables according to reduced costs (dj)

        graph::Separator separator(X);
        if (const auto ucut = separator.Ucut(); ucut.has_value())
        {
            m_model.AddConstraints(std::span{ &*ucut, &*ucut + 1 });
            queue.emplace(currentLowerBound, fixedVariables0, fixedVariables1);
            continue;
        }

        auto fractionalVar = FindFractionalVariable(m_model.GetVariables());

        if (!fractionalVar.has_value())
        {
            bestResult.UpperBound = currentLowerBound;
            bestResult.Paths = CreatePathsFromVariables();

            if (bestResult.LowerBound >= bestResult.UpperBound)
                break;
            else
                continue;
        }

        auto newFixedVariables0 = fixedVariables0;
        newFixedVariables0.push_back(fractionalVar.value());

        auto newFixedVariables1 = fixedVariables1;
        newFixedVariables1.push_back(fractionalVar.value());

        queue.emplace(currentLowerBound, std::move(newFixedVariables0), fixedVariables1);
        queue.emplace(currentLowerBound, fixedVariables0, std::move(newFixedVariables1));
    }

    if (!queue.empty() && std::chrono::steady_clock::now() >= startTime + timeout)
        bestResult.IsTimeoutHit = true;

    bestResult.LowerBound = std::min(bestResult.LowerBound, bestResult.UpperBound);

    return bestResult;
}

std::vector<std::vector<int>> tsplp::MtspModel::CreatePathsFromVariables() const
{
    std::vector<std::vector<int>> paths(A);

    for (size_t a = 0; a < A; ++a)
    {
        paths[a].push_back(m_weightsManager.StartPositions()[a]);
        for (auto i = m_weightsManager.StartPositions()[a]; i != m_weightsManager.EndPositions()[a] || paths[a].size() < 2;)
        {
            for (size_t j = 0; j < N; ++j)
            {
                if (std::abs(X(a, i, j).GetObjectiveValue() - 1.0) < 1.e-10)
                {
                    paths[a].push_back(static_cast<int>(j));
                    i = static_cast<decltype(i)>(j);
                    break;
                }
            }
        }
        assert(paths[a].back() == m_weightsManager.EndPositions()[a]);
    }

    return m_weightsManager.TransformPathsBack(std::move(paths));
}
