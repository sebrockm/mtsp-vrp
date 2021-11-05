#include "MtspModel.hpp"
#include "LinearConstraint.hpp"

#include <cmath>
#include <optional>
#include <queue>
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

tsplp::MtspModel::MtspModel(std::vector<int> startPositions, std::vector<int> endPositions, std::vector<double> weights)
    : m_startPositions(xt::adapt(std::move(startPositions))),
    m_endPositions(xt::adapt(std::move(endPositions))),
    A(std::size(startPositions)),
    N(static_cast<size_t>(std::sqrt(std::size(weights)))),
    m_model(A * N * N),
    W(xt::adapt(std::move(weights), { N, N })),
    X(xt::adapt(m_model.GetVariables(), { A, N, N })),
    m_objective(xt::sum(W* X)())
{
    m_model.SetObjective(m_objective);

    std::vector<LinearConstraint> constraints;
    constraints.reserve(A * N + 3 * A + N * (N - 1) / 2);

    // don't use self referring arcs (entries on diagonal)
    for (size_t a = 0; a < A; ++a)
        for (size_t n = 0; n < N; ++n)
            constraints.emplace_back(X(a, n, n) == 0);

    // special inequalities for start and end nodes
    for (size_t a = 0; a < A; ++a)
    {
        // We write X + 0 instead of X to turn summed up type from Variable to LinearVariableComposition.
        // That is necessary because xtensor initializes the sum with a conversion from 0 to ResultType and we
        // don't provide a conversion from int to Variable, but we do provide one from int to LinearVariableCompositon.
        constraints.emplace_back(xt::sum(xt::view(X + 0, a, m_startPositions[a], xt::all()))() == 1); // arcs out of start nodes
        constraints.emplace_back(xt::sum(xt::view(X + 0, a, xt::all(), m_endPositions[a]))() == 1); // arcs into end nodes
        constraints.emplace_back(X(a, m_endPositions[a], m_startPositions[(a + 1) % A]) == 1); // artificial connections from end to next start
    }

    // inequalities to disallow cycles of length 2
    for (size_t u = 0; u < N; ++u)
        for (size_t v = u + 1; v < N; ++v)
            constraints.emplace_back((xt::sum(xt::view(X + 0, xt::all(), u, v)) + xt::sum(xt::view(X + 0, xt::all(), v, u)))() <= 1);

    m_model.AddConstraints(constraints);
}

tsplp::MtspResult tsplp::MtspModel::BranchAndCutSolve()
{
    MtspResult bestResult{};
    if (m_model.Solve() != Status::Optimal)
        return bestResult;

    bestResult.lowerBound = m_objective.Evaluate();
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
    queue.emplace(bestResult.lowerBound, fixedVariables0, fixedVariables1);

    while (!queue.empty())
    {
        UnfixVariables(fixedVariables0);
        UnfixVariables(fixedVariables1);

        bestResult.lowerBound = queue.top().lowerBound;
        fixedVariables0 = queue.top().fixedVariables0;
        fixedVariables1 = queue.top().fixedVariables1;
        queue.pop();

        FixVariables(fixedVariables0, 0.0);
        FixVariables(fixedVariables1, 1.0);

        if (m_model.Solve() != Status::Optimal)
            continue;

        auto currentLowerBound = std::ceil(m_objective.Evaluate() - 1.e-10);

        // do exploiting here

        bestResult.lowerBound = std::min(bestResult.upperBound, currentLowerBound);
        if (!queue.empty())
            bestResult.lowerBound = std::min(currentLowerBound, queue.top().lowerBound);

        if (bestResult.lowerBound >= bestResult.upperBound)
            break;

        if (currentLowerBound >= bestResult.upperBound)
            continue;

        // fix variables according to reduced costs (dj)

        std::vector<LinearConstraint> violatedConstraints; // find violated constraints here
        if (!violatedConstraints.empty())
        {
            m_model.AddConstraints(violatedConstraints);
            queue.emplace(currentLowerBound, fixedVariables0, fixedVariables1);
            continue;
        }

        auto fractionalVar = FindFractionalVariable(m_model.GetVariables());

        if (!fractionalVar.has_value())
        {
            bestResult.upperBound = currentLowerBound;
            // create paths from variables

            if (bestResult.lowerBound >= bestResult.upperBound)
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

    if (queue.empty())
        bestResult.lowerBound = bestResult.upperBound;

    return bestResult;
}
