#include "MtspModel.hpp"
#include "BranchAndCutQueue.hpp"
#include "ConstraintDeque.hpp"
#include "Heuristics.hpp"
#include "LinearConstraint.hpp"
#include "SeparationAlgorithms.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <stdexcept>
#include <thread>

#include <xtensor/xadapt.hpp>
#include <xtensor/xview.hpp>

namespace
{
    void FixVariables(std::span<tsplp::Variable> variables, double value, tsplp::Model& model)
    {
        for (auto v : variables)
        {
            v.SetLowerBound(value, model);
            v.SetUpperBound(value, model);
        }
    }

    void UnfixVariables(std::span<tsplp::Variable> variables, tsplp::Model& model)
    {
        for (auto v : variables)
        {
            v.SetLowerBound(0.0, model);
            v.SetUpperBound(1.0, model);
        }
    }

    std::optional<tsplp::Variable> FindFractionalVariable(const tsplp::Model& model, double epsilon = 1.e-10)
    {
        std::optional<tsplp::Variable> closest = std::nullopt;
        double minAbs = 1.0;
        for (auto v : model.GetVariables())
        {
            if (epsilon <= v.GetObjectiveValue(model) && v.GetObjectiveValue(model) <= 1.0 - epsilon)
            {
                if (std::abs(v.GetObjectiveValue(model) - 0.5) < minAbs)
                {
                    minAbs = std::abs(v.GetObjectiveValue(model) - 0.5);
                    closest = v;
                    if (minAbs < epsilon)
                        break;
                }
            }
        }

        return closest;
    }
}

tsplp::MtspModel::MtspModel(xt::xtensor<size_t, 1> startPositions, xt::xtensor<size_t, 1> endPositions, xt::xtensor<int, 2> weights, std::chrono::milliseconds timeout)
    : m_endTime(m_startTime + timeout),
    m_weightManager(std::move(weights), std::move(startPositions), std::move(endPositions)),
    A(m_weightManager.A()),
    N(m_weightManager.N()),
    m_model(A * N * N),
    X(xt::adapt(m_model.GetVariables(), { A, N, N })),
    m_objective(xt::sum(m_weightManager.W() * X)())
{
    const auto heuristicTimeout = timeout - std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - m_startTime);
    auto [nearestInsertionPaths, nearestInsertionObjective] = NearestInsertion(m_weightManager.W(), m_weightManager.StartPositions(), m_weightManager.EndPositions(), heuristicTimeout);

    m_bestResult.UpperBound = static_cast<double>(nearestInsertionObjective);
    m_bestResult.Paths = std::move(nearestInsertionPaths);

    if (std::chrono::steady_clock::now() >= m_endTime)
    {
        m_bestResult.IsTimeoutHit = true;
        return;
    }

    m_model.SetObjective(m_objective);

    const auto dependencies = xt::argwhere(equal(m_weightManager.W(), -1));
    const auto D = dependencies.size();

    if (std::chrono::steady_clock::now() >= m_endTime)
    {
        m_bestResult.IsTimeoutHit = true;
        return;
    }

    std::vector<LinearConstraint> constraints;
    const auto numberOfConstraints = A * N + 3 * N + 3 * A + D * (3 * A + 1) + N * (N - 1) / 2;
    constraints.reserve(numberOfConstraints);

    // don't use self referring arcs (entries on diagonal)
    for (size_t a = 0; a < A; ++a)
        for (size_t n = 0; n < N; ++n)
            constraints.emplace_back(X(a, n, n) == 0);

    if (std::chrono::steady_clock::now() >= m_endTime)
    {
        m_bestResult.IsTimeoutHit = true;
        return;
    }

    // degree inequalities
    for (size_t n = 0; n < N; ++n)
    {
        constraints.emplace_back(xt::sum(xt::view(X + 0, xt::all(), xt::all(), n))() == 1);
        constraints.emplace_back(xt::sum(xt::view(X + 0, xt::all(), n, xt::all()))() == 1);

        // each node must be entered and left by the same agent (except start nodes which are artificially entered by previous agent)
        if (std::find(m_weightManager.StartPositions().begin(), m_weightManager.StartPositions().end(), n) == m_weightManager.StartPositions().end())
        {
            for (size_t a = 0; a < A; ++a)
                constraints.emplace_back(xt::sum(xt::view(X + 0, a, xt::all(), n))() == xt::sum(xt::view(X + 0, a, n, xt::all()))());
        }
    }

    if (std::chrono::steady_clock::now() >= m_endTime)
    {
        m_bestResult.IsTimeoutHit = true;
        return;
    }

    // special inequalities for start and end nodes
    for (size_t a = 0; a < A; ++a)
    {
        // We write X + 0 instead of X to turn summed up type from Variable to LinearVariableComposition.
        // That is necessary because xtensor initializes the sum with a conversion from 0 to ResultType and we
        // don't provide a conversion from int to Variable, but we do provide one from int to LinearVariableCompositon.
        constraints.emplace_back(xt::sum(xt::view(X + 0, a, m_weightManager.StartPositions()[a], xt::all()))() == 1); // arcs out of start nodes
        constraints.emplace_back(xt::sum(xt::view(X + 0, a, xt::all(), m_weightManager.EndPositions()[a]))() == 1); // arcs into end nodes
        constraints.emplace_back(X(a, m_weightManager.EndPositions()[a], m_weightManager.StartPositions()[(a + 1) % A]) == 1); // artificial connections from end to next start
    }

    for (const auto [v, u] : dependencies)
    {
        assert(std::find(m_weightManager.StartPositions().begin(), m_weightManager.StartPositions().end(), v) == m_weightManager.StartPositions().end());
        assert(std::find(m_weightManager.EndPositions().begin(), m_weightManager.EndPositions().end(), u) == m_weightManager.EndPositions().end());

        if (A == 1)
        {
            [[maybe_unused]] const auto s = m_weightManager.StartPositions()[0];
            [[maybe_unused]] const auto e = m_weightManager.EndPositions()[0];
            assert(s != u || e != v); // TODO: ensure that W[e, s] == 0, even though W[e, s] == -1 would be plausible. Arc (e, s) must be used in case A == 1
        }

        constraints.emplace_back(xt::sum(xt::view(X + 0, xt::all(), v, u))() == 0); // reverse edge of dependency must not be used

        for (size_t a = 0; a < A; ++a)
            constraints.emplace_back(xt::sum(xt::view(X + 0, a, u, xt::all()))() == xt::sum(xt::view(X + 0, a, xt::all(), v))()); // require the same agent to visit dependent nodes

        for (const auto s : m_weightManager.StartPositions())
        {
            if (s != u)
                constraints.emplace_back(xt::sum(xt::view(X + 0, xt::all(), s, v))() == 0); // u->v, so startPosition->v is not possible
        }

        for (const auto e : m_weightManager.EndPositions())
        {
            if (e != v)
                constraints.emplace_back(xt::sum(xt::view(X + 0, xt::all(), u, e))() == 0); // u->v, so u->endPosition is not possible
        }

        if (std::chrono::steady_clock::now() >= m_endTime)
        {
            m_bestResult.IsTimeoutHit = true;
            return;
        }
    }

    // inequalities to disallow cycles of length 2
    for (size_t u = 0; u < N; ++u)
    {
        for (size_t v = u + 1; v < N; ++v)
            constraints.emplace_back((xt::sum(xt::view(X + 0, xt::all(), u, v)) + xt::sum(xt::view(X + 0, xt::all(), v, u)))() <= 1);

        if (std::chrono::steady_clock::now() >= m_endTime)
        {
            m_bestResult.IsTimeoutHit = true;
            return;
        }
    }

    m_model.AddConstraints(constraints);
}

tsplp::MtspResult tsplp::MtspModel::BranchAndCutSolve(std::optional<size_t> noOfThreads)
{
    using namespace std::chrono_literals;

    const auto threadCount = noOfThreads && *noOfThreads > 0 ? *noOfThreads : std::thread::hardware_concurrency();

    auto remainingTime = std::chrono::duration_cast<std::chrono::milliseconds>(m_endTime - std::chrono::steady_clock::now());

    if (remainingTime <= 0ms)
    {
        m_bestResult.IsTimeoutHit = true;
        return m_bestResult;
    }

    BranchAndCutQueue queue;
    ConstraintDeque constraints(threadCount);

    const auto threadLoop = [&](const size_t threadId)
    {
        auto model = m_model;
        graph::Separator separator(X, m_weightManager, model);

        std::vector<Variable> fixedVariables0{};
        std::vector<Variable> fixedVariables1{};

        while (true)
        {
            UnfixVariables(fixedVariables0, model);
            UnfixVariables(fixedVariables1, model);

            auto top = queue.Pop(threadId);
            if (!top.has_value())
                break;

            if (std::chrono::steady_clock::now() >= m_endTime)
            {
                queue.ClearAll();
                break;
            }

            fixedVariables0 = std::move(top->FixedVariables0);
            fixedVariables1 = std::move(top->FixedVariables1);

            FixVariables(fixedVariables0, 0.0, model);
            FixVariables(fixedVariables1, 1.0, model);

            constraints.PopToModel(threadId, model);

            remainingTime = std::chrono::duration_cast<std::chrono::milliseconds>(m_endTime - std::chrono::steady_clock::now());

            if (model.Solve(remainingTime) != Status::Optimal)
            {
                queue.NotifyNodeDone(threadId);
                continue;
            }

            const auto currentLowerBound = std::ceil(m_objective.Evaluate(model) - 1.e-10);
            queue.UpdateCurrentLowerBound(threadId, currentLowerBound);

            // do exploiting here

            {
                std::unique_lock lock{ m_bestResultMutex };

                const auto threadLowerBound = std::min(m_bestResult.UpperBound, currentLowerBound);

                m_bestResult.LowerBound = std::min(threadLowerBound, queue.GetLowerBound().value_or(threadLowerBound));

                if (m_bestResult.LowerBound >= m_bestResult.UpperBound)
                {
                    queue.ClearAll();
                    break;
                }

                if (currentLowerBound >= m_bestResult.UpperBound)
                {
                    queue.NotifyNodeDone(threadId);
                    continue;
                }
            }

            // fix variables according to reduced costs (dj)

            if (auto ucut = separator.Ucut(); ucut.has_value())
            {
                constraints.Push(std::move(*ucut));
                queue.Push(currentLowerBound, fixedVariables0, fixedVariables1);
                queue.NotifyNodeDone(threadId);
                continue;
            }

            if (auto pisigma = separator.PiSigma(); pisigma.has_value())
            {
                constraints.Push(std::move(*pisigma));
                queue.Push(currentLowerBound, fixedVariables0, fixedVariables1);
                queue.NotifyNodeDone(threadId);
                continue;
            }

            if (auto pi = separator.Pi(); pi.has_value())
            {
                constraints.Push(std::move(*pi));
                queue.Push(currentLowerBound, fixedVariables0, fixedVariables1);
                queue.NotifyNodeDone(threadId);
                continue;
            }

            if (auto sigma = separator.Sigma(); sigma.has_value())
            {
                constraints.Push(std::move(*sigma));
                queue.Push(currentLowerBound, fixedVariables0, fixedVariables1);
                queue.NotifyNodeDone(threadId);
                continue;
            }

            const auto fractionalVar = FindFractionalVariable(model);

            if (!fractionalVar.has_value())
            {
                std::unique_lock lock{ m_bestResultMutex };

                if (currentLowerBound >= m_bestResult.UpperBound) // another thread may have updated UpperBound since the last check
                {
                    queue.NotifyNodeDone(threadId);
                    continue;
                }

                m_bestResult.UpperBound = currentLowerBound;
                m_bestResult.Paths = CreatePathsFromVariables(model);

                if (m_bestResult.LowerBound >= m_bestResult.UpperBound)
                {
                    queue.ClearAll();
                    break;
                }
                else
                {
                    queue.NotifyNodeDone(threadId);
                    continue;
                }
            }

            queue.PushBranch(currentLowerBound, fixedVariables0, fixedVariables1, fractionalVar.value());
            queue.NotifyNodeDone(threadId);
        }
    };

    std::vector<std::thread> threads;
    for (size_t i = 0; i < threadCount; ++i)
        threads.emplace_back(threadLoop, i);

    for (auto& thread : threads)
        thread.join();

    if (m_bestResult.LowerBound < m_bestResult.UpperBound && std::chrono::steady_clock::now() >= m_endTime)
        m_bestResult.IsTimeoutHit = true;

    m_bestResult.LowerBound = std::min(m_bestResult.LowerBound, m_bestResult.UpperBound);

    return m_bestResult;
}

std::vector<std::vector<size_t>> tsplp::MtspModel::CreatePathsFromVariables(const Model& model) const
{
    std::vector<std::vector<size_t>> paths(A);

    for (size_t a = 0; a < A; ++a)
    {
        paths[a].push_back(m_weightManager.StartPositions()[a]);

        for (size_t i = 1; i < N; ++i)
        {
            for (size_t n = 0; n < N; ++n)
            {
                if (X(a, paths[a].back(), n).GetObjectiveValue(model) > 1 - 1.e-10)
                {
                    paths[a].push_back(n);
                    break;
                }
            }

            if (paths[a].back() == m_weightManager.EndPositions()[a])
                break;
        }
        assert(paths[a].back() == m_weightManager.EndPositions()[a]);
    }

    return m_weightManager.TransformPathsBack(std::move(paths));
}
