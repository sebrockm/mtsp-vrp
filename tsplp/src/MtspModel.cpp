#include "MtspModel.hpp"

#include "BranchAndCutQueue.hpp"
#include "ConstraintDeque.hpp"
#include "Heuristics.hpp"
#include "LinearConstraint.hpp"
#include "SeparationAlgorithms.hpp"

#include <xtensor/xadapt.hpp>
#include <xtensor/xview.hpp>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <stdexcept>
#include <thread>

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

std::optional<tsplp::Variable> FindFractionalVariable(
    const tsplp::Model& model, double epsilon = 1.e-10)
{
    std::optional<tsplp::Variable> closest = std::nullopt;
    double minAbs = 1.0;
    for (auto v : model.GetVariables())
    {
        if (epsilon <= v.GetObjectiveValue(model) && v.GetObjectiveValue(model) <= 1.0 - epsilon)
        {
            if (const auto abs = std::abs(v.GetObjectiveValue(model) - 0.5); abs < minAbs)
            {
                minAbs = abs;
                closest = v;
                if (minAbs < epsilon)
                    break;
            }
        }
    }

    return closest;
}
}

tsplp::MtspModel::MtspModel(
    xt::xtensor<size_t, 1> startPositions, xt::xtensor<size_t, 1> endPositions,
    xt::xtensor<int, 2> weights, std::chrono::milliseconds timeout)
    : m_endTime(m_startTime + timeout)
    , m_weightManager(std::move(weights), std::move(startPositions), std::move(endPositions))
    , A(m_weightManager.A())
    , N(m_weightManager.N())
    , m_model(A * N * N)
    , X(xt::adapt(m_model.GetVariables(), { A, N, N }))
    , m_objective(xt::sum(m_weightManager.W() * X)())
{
    const auto heuristicTimeout = timeout
        - std::chrono::duration_cast<std::chrono::milliseconds>(
                                      std::chrono::steady_clock::now() - m_startTime);
    auto [nearestInsertionPaths, nearestInsertionObjective] = NearestInsertion(
        m_weightManager.W(), m_weightManager.StartPositions(), m_weightManager.EndPositions(),
        m_weightManager.Dependencies(), heuristicTimeout);

    m_bestResult.UpperBound = nearestInsertionObjective;
    auto [twoOptedPaths, twoOptImprovement] = TwoOptPaths(
        std::move(nearestInsertionPaths), m_weightManager.W(), m_weightManager.Dependencies());

    m_bestResult.Paths = std::move(twoOptedPaths);
    m_bestResult.UpperBound -= twoOptImprovement;

    if (std::chrono::steady_clock::now() >= m_endTime)
    {
        m_bestResult.IsTimeoutHit = true;
        return;
    }

    m_model.SetObjective(m_objective);

    if (std::chrono::steady_clock::now() >= m_endTime)
    {
        m_bestResult.IsTimeoutHit = true;
        return;
    }

    std::vector<LinearConstraint> constraints;

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

        // each node must be entered and left by the same agent (except start nodes which are
        // artificially entered by previous agent)
        if (std::find(
                m_weightManager.StartPositions().begin(), m_weightManager.StartPositions().end(), n)
            == m_weightManager.StartPositions().end())
        {
            for (size_t a = 0; a < A; ++a)
                constraints.emplace_back(
                    xt::sum(xt::view(X + 0, a, xt::all(), n))()
                    == xt::sum(xt::view(X + 0, a, n, xt::all()))());
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
        // We write X + 0 instead of X to turn summed up type from Variable to
        // LinearVariableComposition. That is necessary because xtensor initializes the sum with a
        // conversion from 0 to ResultType and we don't provide a conversion from int to Variable,
        // but we do provide one from int to LinearVariableCompositon.

        // arcs out of start nodes
        constraints.emplace_back(
            xt::sum(xt::view(X + 0, a, m_weightManager.StartPositions()[a], xt::all()))() == 1);

        // arcs into end nodes
        constraints.emplace_back(
            xt::sum(xt::view(X + 0, a, xt::all(), m_weightManager.EndPositions()[a]))() == 1);

        // artificial connections from end to next start
        constraints.emplace_back(
            X(a, m_weightManager.EndPositions()[a], m_weightManager.StartPositions()[(a + 1) % A])
            == 1);
    }

    for (const auto& [u, v] : m_weightManager.Dependencies().GetArcs())
    {
        assert(
            std::find(
                m_weightManager.StartPositions().begin(), m_weightManager.StartPositions().end(), v)
            == m_weightManager.StartPositions().end());
        assert(
            std::find(
                m_weightManager.EndPositions().begin(), m_weightManager.EndPositions().end(), u)
            == m_weightManager.EndPositions().end());

        if (A == 1)
        {
            [[maybe_unused]] const auto s = m_weightManager.StartPositions()[0];
            [[maybe_unused]] const auto e = m_weightManager.EndPositions()[0];

            // TODO: ensure that W[e, s] == 0, even though W[e, s] == -1 would be plausible. Arc (e,
            // s) must be used in case A == 1
            assert(s != u || e != v);
        }

        // reverse edge of dependency must not be used
        constraints.emplace_back(xt::sum(xt::view(X + 0, xt::all(), v, u))() == 0);

        // require the same agent to visit dependent nodes
        for (size_t a = 0; a < A; ++a)
            constraints.emplace_back(
                xt::sum(xt::view(X + 0, a, u, xt::all()))()
                == xt::sum(xt::view(X + 0, a, xt::all(), v))());

        for (const auto s : m_weightManager.StartPositions())
        {
            // u->v, so startPosition->v is not possible
            if (s != u)
                constraints.emplace_back(xt::sum(xt::view(X + 0, xt::all(), s, v))() == 0);
        }

        for (const auto e : m_weightManager.EndPositions())
        {
            // u->v, so u->endPosition is not possible
            if (e != v)
                constraints.emplace_back(xt::sum(xt::view(X + 0, xt::all(), u, e))() == 0);
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
            constraints.emplace_back(
                (xt::sum(xt::view(X + 0, xt::all(), u, v))
                 + xt::sum(xt::view(X + 0, xt::all(), v, u)))()
                <= 1);

        if (std::chrono::steady_clock::now() >= m_endTime)
        {
            m_bestResult.IsTimeoutHit = true;
            return;
        }
    }

    m_model.AddConstraints(cbegin(constraints), cend(constraints));
}

tsplp::MtspResult tsplp::MtspModel::BranchAndCutSolve(
    std::optional<size_t> noOfThreads,
    std::function<void(const xt::xtensor<double, 3>&)> fractionalCallback)
{
    using namespace std::chrono_literals;

    const auto threadCount
        = noOfThreads && *noOfThreads > 0 ? *noOfThreads : std::thread::hardware_concurrency();

    auto remainingTime = std::chrono::duration_cast<std::chrono::milliseconds>(
        m_endTime - std::chrono::steady_clock::now());

    if (remainingTime <= 0ms)
    {
        m_bestResult.IsTimeoutHit = true;
        return m_bestResult;
    }

    auto callbackMutex = [&]() -> std::optional<std::mutex>
    {
        if (fractionalCallback != nullptr)
            return std::make_optional<std::mutex>();
        return std::nullopt;
    }();

    BranchAndCutQueue queue;
    ConstraintDeque constraints(threadCount);

    const auto threadLoop = [&](const size_t threadId)
    {
        auto model = m_model;
        graph::Separator separator(X, m_weightManager, model);

        std::vector<Variable> fixedVariables0 {};
        std::vector<Variable> fixedVariables1 {};

        while (true)
        {
            // unfix variables from previous loop iteration to get a clean model
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

            remainingTime = std::chrono::duration_cast<std::chrono::milliseconds>(
                m_endTime - std::chrono::steady_clock::now());

            if (model.Solve(remainingTime) != Status::Optimal)
            {
                queue.NotifyNodeDone(threadId);
                continue;
            }

            const auto currentLowerBound = std::ceil(m_objective.Evaluate(model) - 1.e-10);
            queue.UpdateCurrentLowerBound(threadId, currentLowerBound);

            const auto currentUpperBound = [this]
            {
                std::unique_lock lock { m_bestResultMutex };
                return m_bestResult.UpperBound;
            }();

            if (2.5 * currentLowerBound > currentUpperBound || fractionalCallback != nullptr)
            {
                const xt::xtensor<double, 3> fractionalValues
                    = xt::vectorize([&](Variable v) { return v.GetObjectiveValue(model); })(X);

                if (fractionalCallback != nullptr)
                {
                    std::unique_lock lock { *callbackMutex };
                    fractionalCallback(m_weightManager.TransformTensorBack(fractionalValues));
                }

                // don't exploit if there isn't a reasonable chance, 2.5 might be adjusted
                if (2.5 * currentLowerBound > currentUpperBound)
                {
                    remainingTime = std::chrono::duration_cast<std::chrono::milliseconds>(
                        m_endTime - std::chrono::steady_clock::now());

                    auto [exploitedPaths, exploitedObjective] = ExploitFractionalSolution(
                        fractionalValues, m_weightManager.W(), m_weightManager.StartPositions(),
                        m_weightManager.EndPositions(), m_weightManager.Dependencies(),
                        remainingTime);

                    if (!exploitedPaths.empty())
                    {
                        auto [twoOptedPaths, twoOptImprovement] = TwoOptPaths(
                            std::move(exploitedPaths), m_weightManager.W(),
                            m_weightManager.Dependencies());
                        exploitedObjective -= twoOptImprovement;

                        std::unique_lock lock { m_bestResultMutex };

                        if (exploitedObjective < m_bestResult.UpperBound)
                        {
                            m_bestResult.UpperBound = exploitedObjective;
                            m_bestResult.Paths = std::move(twoOptedPaths);
                        }
                    }
                }
            }

            {
                std::unique_lock lock { m_bestResultMutex };

                const auto threadLowerBound = std::min(m_bestResult.UpperBound, currentLowerBound);

                m_bestResult.LowerBound
                    = std::min(threadLowerBound, queue.GetLowerBound().value_or(threadLowerBound));

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

            // fix variables according to reduced costs
            for (auto v : model.GetVariables())
            {
                if (v.GetLowerBound(model) == 0.0 && v.GetUpperBound(model) == 1.0)
                {
                    if (v.GetObjectiveValue(model) < 1.e-10
                        && currentLowerBound + v.GetReducedCosts(model)
                            >= currentUpperBound + 1.e-10)
                    {
                        fixedVariables0.push_back(v);
                    }
                    else if (
                        v.GetObjectiveValue(model) > 1 - 1.e-10
                        && currentLowerBound - v.GetReducedCosts(model)
                            >= currentUpperBound + 1.e-10)
                    {
                        fixedVariables1.push_back(v);
                    }
                }
            }

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
                std::unique_lock lock { m_bestResultMutex };

                if (currentLowerBound >= m_bestResult.UpperBound) // another thread may have updated
                                                                  // UpperBound since the last check
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

            queue.PushBranch(
                currentLowerBound, fixedVariables0, fixedVariables1, fractionalVar.value());
            queue.NotifyNodeDone(threadId);
        }
    };

    std::vector<std::thread> threads;
    for (size_t i = 0; i < threadCount; ++i)
        threads.emplace_back(threadLoop, i);

    for (auto& thread : threads)
        thread.join();

    if (m_bestResult.LowerBound < m_bestResult.UpperBound
        && std::chrono::steady_clock::now() >= m_endTime)
        m_bestResult.IsTimeoutHit = true;

    m_bestResult.LowerBound = std::min(m_bestResult.LowerBound, m_bestResult.UpperBound);

    return m_bestResult;
}

std::vector<std::vector<size_t>> tsplp::MtspModel::CreatePathsFromVariables(
    const Model& model) const
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
