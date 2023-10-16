#include "MtspModel.hpp"

#include "BranchAndCutQueue.hpp"
#include "ConstraintDeque.hpp"
#include "Heuristics.hpp"
#include "LinearConstraint.hpp"
#include "SeparationAlgorithms.hpp"

#include <xtensor/xadapt.hpp>
#include <xtensor/xvectorize.hpp>
#include <xtensor/xview.hpp>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <thread>

namespace
{
void FixVariables(std::span<const tsplp::Variable> variables, double value, tsplp::Model& model)
{
    for (const auto v : variables)
        v.Fix(value, model);
}

void UnfixVariables(std::span<const tsplp::Variable> variables, tsplp::Model& model)
{
    for (const auto v : variables)
        v.Unfix(model);
}

std::optional<tsplp::Variable> FindFractionalVariable(
    const tsplp::Model& model, double epsilon = 1.e-10)
{
    std::optional<tsplp::Variable> closest = std::nullopt;
    double minAbs = 1.0;
    for (const auto v : model.GetBinaryVariables())
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
    xt::xtensor<double, 2> weights, OptimizationMode optimizationMode,
    std::chrono::milliseconds timeout, std::string name)
    : m_endTime(m_startTime + timeout)
    , m_weightManager(std::move(weights), std::move(startPositions), std::move(endPositions))
    , m_optimizationMode(optimizationMode)
    , A(m_weightManager.A())
    , N(m_weightManager.N())
    , m_name(std::move(name))
{
    if (m_optimizationMode != OptimizationMode::Sum && m_optimizationMode != OptimizationMode::Max)
        return;

    CreateInitialResult();

    if (std::chrono::steady_clock::now() >= m_endTime)
    {
        m_bestResult.SetTimeoutHit();
        return;
    }

    m_model = Model(A * N * N);
    X = xt::adapt(
        m_model.GetBinaryVariables().data(), A * N * N, xt::no_ownership {},
        std::array { A, N, N });

    const auto maxVariable = m_optimizationMode == OptimizationMode::Max
        ? std::make_optional(m_model.AddVariable(0.0, std::numeric_limits<double>::max()))
        : std::nullopt;

    m_objective = CreateObjective(m_weightManager.W(), X, maxVariable);
    m_model.AddConstraints(
        cbegin(m_objective.AdditionalConstraints), cend(m_objective.AdditionalConstraints));

    m_model.SetObjective(m_objective.Objective);

    std::vector<LinearConstraint> constraints;

    // don't use self referring arcs (entries on diagonal)
    for (size_t a = 0; a < A; ++a)
    {
        for (size_t n = 0; n < N; ++n)
            constraints.emplace_back(X(a, n, n) == 0);
    }

    if (std::chrono::steady_clock::now() >= m_endTime)
    {
        m_bestResult.SetTimeoutHit();
        return;
    }

    // degree inequalities
    for (size_t n = 0; n < N; ++n)
    {
        LinearVariableComposition incoming;
        for (size_t a = 0; a < A; ++a)
        {
            for (size_t m = 0; m < N; ++m)
                incoming += X(a, m, n);
        }
        constraints.emplace_back(std::move(incoming) == 1);

        LinearVariableComposition outgoing;
        for (size_t a = 0; a < A; ++a)
        {
            for (size_t m = 0; m < N; ++m)
                outgoing += X(a, n, m);
        }
        constraints.emplace_back(std::move(outgoing) == 1);

        // each node must be entered and left by the same agent (except start nodes which are
        // artificially entered by previous agent)
        if (std::find(
                m_weightManager.StartPositions().begin(), m_weightManager.StartPositions().end(), n)
            == m_weightManager.StartPositions().end())
        {
            for (size_t a = 0; a < A; ++a)
            {
                LinearVariableComposition incomingA;
                for (size_t m = 0; m < N; ++m)
                    incomingA += X(a, m, n);

                LinearVariableComposition outgoingA;
                for (size_t m = 0; m < N; ++m)
                    outgoingA += X(a, n, m);

                constraints.emplace_back(std::move(incomingA) == std::move(outgoingA));
            }
        }
    }

    if (std::chrono::steady_clock::now() >= m_endTime)
    {
        m_bestResult.SetTimeoutHit();
        return;
    }

    // special inequalities for start and end nodes
    for (size_t a = 0; a < A; ++a)
    {
        const auto s = m_weightManager.StartPositions()[a];
        const auto e = m_weightManager.EndPositions()[a];

        LinearVariableComposition outOfStart;
        for (size_t v = 0; v < N; ++v)
            outOfStart += X(a, s, v);
        constraints.push_back(std::move(outOfStart) == 1);

        LinearVariableComposition intoEnd;
        for (size_t u = 0; u < N; ++u)
            intoEnd += X(a, u, e);
        constraints.push_back(std::move(intoEnd) == 1);

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

            assert(std::make_pair(u, v) != std::make_pair(e, s));
        }

        // Reverse edge of dependency must not be used.
        // Exception: in the case of A == 1, if there is a dependency from start to end node,
        // the reverse edge must be used to complete a full TSP cycle.
        // (see section "artificial connections from end to next start" above)
        if (A > 1 || u != m_weightManager.StartPositions()[0]
            || v != m_weightManager.EndPositions()[0])
        {
            LinearVariableComposition reverseEdge;
            for (size_t a = 0; a < A; ++a)
                reverseEdge += X(a, v, u);
            constraints.push_back(std::move(reverseEdge) == 0);
        }

        // require the same agent to visit dependent nodes
        if (A > 1)
        {
            for (size_t a = 0; a < A; ++a)
            {
                LinearVariableComposition outgoing;
                for (size_t n = 0; n < N; ++n)
                    outgoing += X(a, u, n);

                LinearVariableComposition incoming;
                for (size_t n = 0; n < N; ++n)
                    incoming += X(a, n, v);

                constraints.push_back(std::move(outgoing) == std::move(incoming));
            }
        }

        for (const auto s : m_weightManager.StartPositions())
        {
            // u->v, so startPosition->v is not possible
            if (s != u)
            {
                LinearVariableComposition startToV;
                for (size_t a = 0; a < A; ++a)
                    startToV += X(a, s, v);
                constraints.push_back(std::move(startToV) == 0);
            }
        }

        for (const auto e : m_weightManager.EndPositions())
        {
            // u->v, so u->endPosition is not possible
            if (e != v)
            {
                LinearVariableComposition uToEnd;
                for (size_t a = 0; a < A; ++a)
                    uToEnd += X(a, u, e);
                constraints.push_back(std::move(uToEnd) == 0);
            }
        }

        if (std::chrono::steady_clock::now() >= m_endTime)
        {
            m_bestResult.SetTimeoutHit();
            return;
        }
    }

    // inequalities to disallow cycles of length 2
    for (size_t u = 0; u < N; ++u)
    {
        for (size_t v = u + 1; v < N; ++v)
        {
            LinearVariableComposition cycle;
            for (size_t a = 0; a < A; ++a)
            {
                cycle += X(a, u, v);
                cycle += X(a, v, u);
            }
            constraints.push_back(std::move(cycle) <= 1);
        }

        if (std::chrono::steady_clock::now() >= m_endTime)
        {
            m_bestResult.SetTimeoutHit();
            return;
        }
    }

    m_model.AddConstraints(cbegin(constraints), cend(constraints));
}

void tsplp::MtspModel::BranchAndCutSolve(
    std::optional<size_t> noOfThreads,
    std::function<void(const xt::xtensor<double, 3>&)> fractionalCallback)
{
    using namespace std::chrono_literals;

    const auto threadCount
        = noOfThreads && *noOfThreads > 0 ? *noOfThreads : std::thread::hardware_concurrency();

    if (std::chrono::steady_clock::now() >= m_endTime)
    {
        m_bestResult.SetTimeoutHit();
        return;
    }

    auto callbackMutex = [&]() -> std::optional<std::mutex>
    {
        if (fractionalCallback != nullptr)
            return std::make_optional<std::mutex>();
        return std::nullopt;
    }();

    BranchAndCutQueue queue(threadCount);
    queue.Push(0, {}, {});
    ConstraintDeque constraints(threadCount);

    const auto threadLoop = [&](const size_t threadId)
    {
        auto model = m_model;
        const graph::Separator separator(X, m_weightManager, model);

        std::vector<Variable> fixedVariables0 {};
        std::vector<Variable> fixedVariables1 {};

        while (true)
        {
            const auto initialBounds = m_bestResult.UpdateLowerBound(queue.GetLowerBound());

            if (initialBounds.Lower >= initialBounds.Upper)
            {
                queue.ClearAll();
                break;
            }

            if (std::chrono::steady_clock::now() >= m_endTime)
            {
                queue.ClearAll();
                break;
            }

            // unfix variables from previous loop iteration to get a clean model
            UnfixVariables(fixedVariables0, model);
            UnfixVariables(fixedVariables1, model);

            auto top = queue.Pop(threadId);
            if (!top.has_value())
            {
                break;
            }

            auto& [sdata, nodeDoneNotifier] = *top;

            if (sdata.IsResult)
            {
                const auto globalLowerBound
                    = m_bestResult.UpdateLowerBound(queue.GetLowerBound()).Lower;
                if (sdata.LowerBound > globalLowerBound)
                {
                    // As this was just popped but is not the global LB, this means other threads
                    // are currently working on smaller LBs. Push it back to be reevaluated later.
                    queue.PushResult(sdata.LowerBound);
                }

                continue;
            }

            fixedVariables0 = std::move(sdata.FixedVariables0);
            fixedVariables1 = std::move(sdata.FixedVariables1);

            FixVariables(fixedVariables0, 0.0, model);
            FixVariables(fixedVariables1, 1.0, model);

            constraints.PopToModel(threadId, model);

            const auto solutionStatus = model.Solve(m_endTime);
            switch (solutionStatus)
            {
            case Status::Unbounded:
                throw std::logic_error(
                    m_name
                    + ": LP solution is unbounded. This must not happen. Maybe some "
                      "constraints are missing.");
            case Status::Error:
                throw std::logic_error(m_name + ": Unexpected error happened while solving LP.");
            case Status::Timeout: // timeout will be handled at the beginning of the next iteration
            case Status::Infeasible: // fixation of some variable makes this infeasible, skip it
                continue;
            case Status::Optimal:
                break;
            }

            const auto currentLowerBound
                = std::ceil(m_objective.Objective.Evaluate(model) - 1.e-10);

            queue.UpdateCurrentLowerBound(threadId, currentLowerBound);

            auto currentUpperBound = m_bestResult.UpdateLowerBound(queue.GetLowerBound()).Upper;

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
                    currentUpperBound = ExploitFractionalSolution(fractionalValues);
            }

            // currentLowerBound is not necessarily the global LB, but either way there is no need
            // trying to improve it further
            if (currentLowerBound >= currentUpperBound)
            {
                queue.PushResult(currentLowerBound);
                continue;
            }

            // fix variables according to reduced costs
            for (auto v : model.GetBinaryVariables())
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

                        const auto recursivelyFixed0 = CalculateRecursivelyFixableVariables(v);
                        fixedVariables0.insert(
                            fixedVariables0.end(), recursivelyFixed0.begin(),
                            recursivelyFixed0.end());
                    }
                }
            }

            if (auto ucut = separator.Ucut(); ucut.has_value())
            {
                constraints.Push(std::move(*ucut));
                queue.Push(currentLowerBound, fixedVariables0, fixedVariables1);
                continue;
            }

            if (auto pisigma = separator.PiSigma(); pisigma.has_value())
            {
                constraints.Push(std::move(*pisigma));
                queue.Push(currentLowerBound, fixedVariables0, fixedVariables1);
                continue;
            }

            if (auto pi = separator.Pi(); pi.has_value())
            {
                constraints.Push(std::move(*pi));
                queue.Push(currentLowerBound, fixedVariables0, fixedVariables1);
                continue;
            }

            if (auto sigma = separator.Sigma(); sigma.has_value())
            {
                constraints.Push(std::move(*sigma));
                queue.Push(currentLowerBound, fixedVariables0, fixedVariables1);
                continue;
            }

            if (auto combs = separator.TwoMatching(); !combs.empty())
            {
                constraints.Push(
                    std::make_move_iterator(combs.begin()), std::make_move_iterator(combs.end()));
                queue.Push(currentLowerBound, fixedVariables0, fixedVariables1);
                continue;
            }

            const auto fractionalVar = FindFractionalVariable(model);

            // The fractional solution happens to be all integer and no constraint violations have
            // been found above, so this is a solution for the actual problem.
            if (!fractionalVar.has_value())
            {
                // another thread may have updated the upper bound since the last check
                if (currentLowerBound < m_bestResult.GetBounds().Upper)
                {
                    m_bestResult.UpdateUpperBound(
                        currentLowerBound, CreatePathsFromVariables(model));
                }

                queue.PushResult(currentLowerBound);
                continue;
            }

            // As a last resort, split the problem on a fractional variable
            auto recursivelyFixed0 = CalculateRecursivelyFixableVariables(fractionalVar.value());
            queue.PushBranch(
                currentLowerBound, fixedVariables0, fixedVariables1, fractionalVar.value(),
                std::move(recursivelyFixed0));
        }
    };

    std::vector<std::thread> threads;
    for (size_t i = 1; i < threadCount; ++i)
        threads.emplace_back(threadLoop, i);

    threadLoop(0);

    for (auto& thread : threads)
        thread.join();

    const auto [lowerBound, upperBound] = m_bestResult.GetBounds();
    assert(lowerBound <= upperBound);

    if (lowerBound < upperBound)
    {
        if (std::chrono::steady_clock::now() < m_endTime)
        {
            throw std::logic_error(
                m_name + ": Logic Error: Timeout not reached, but no optimal solution found.");
        }
        m_bestResult.SetTimeoutHit();
    }
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

std::vector<tsplp::Variable> tsplp::MtspModel::CalculateRecursivelyFixableVariables(
    Variable var) const
{
    // agent a uses edge (u, v)
    const auto v = var.GetId() % N;
    const auto u = (var.GetId() / N) % N;
    const auto a = (var.GetId() / N / N);

    const auto isUStart
        = std::find(
              m_weightManager.StartPositions().begin(), m_weightManager.StartPositions().end(), u)
        != m_weightManager.StartPositions().end();
    const auto isVEnd
        = std::find(m_weightManager.EndPositions().begin(), m_weightManager.EndPositions().end(), v)
        != m_weightManager.EndPositions().end();

    std::vector<Variable> result;

    for (size_t aa = 0; aa < A; ++aa)
    {
        // no other agent can use (u, v)
        if (aa != a)
            result.emplace_back(aa * N * N + u * N + v);

        // all other edges leaving u, no matter which agent, cannot be used
        for (size_t vv = 0; vv < N; ++vv)
        {
            if (vv != v)
                result.emplace_back(aa * N * N + u * N + vv);
        }

        // all other edges entering v, no matter which agent, cannot be used
        for (size_t uu = 0; uu < N; ++uu)
        {
            if (uu != u)
                result.emplace_back(aa * N * N + uu * N + v);
        }

        // edges entering u with a different agent cannot be used
        // if u is a start node, agents are fixed anyway
        if (!isUStart && aa != a)
        {
            for (size_t w = 0; w < N; ++w)
                result.emplace_back(aa * N * N + w * N + u);
        }

        // edges leaving v with a different agent cannot be used
        // if v is an end node, agents are fixed anyway
        if (!isVEnd && aa != a)
        {
            for (size_t w = 0; w < N; ++w)
                result.emplace_back(aa * N * N + v * N + w);
        }

        if (aa != a)
        {
            // dependees of v cannot use another agent
            const auto dependees = m_weightManager.Dependencies().GetOutgoingSpan(v);
            for (const auto d : dependees)
            {
                if (std::find(
                        m_weightManager.EndPositions().begin(),
                        m_weightManager.EndPositions().end(), d)
                    == m_weightManager.EndPositions().end())
                {
                    for (size_t w = 0; w < N; ++w)
                    {
                        // edges into d must use agent a
                        result.emplace_back(aa * N * N + w * N + d);

                        // edges out of d must use agent a
                        result.emplace_back(aa * N * N + d * N + w);
                    }
                }
            }

            // dependers of u cannot use another agent
            const auto dependers = m_weightManager.Dependencies().GetIncomingSpan(u);
            for (const auto d : dependers)
            {
                if (std::find(
                        m_weightManager.StartPositions().begin(),
                        m_weightManager.StartPositions().end(), d)
                    == m_weightManager.StartPositions().end())
                {
                    for (size_t w = 0; w < N; ++w)
                    {
                        // edges into d must use agent a
                        result.emplace_back(aa * N * N + w * N + d);

                        // edges out of d must use agent a
                        result.emplace_back(aa * N * N + d * N + w);
                    }
                }
            }
        }
    }

    return result;
}

void tsplp::MtspModel::CreateInitialResult()
{
    auto [nearestInsertionPaths, nearestInsertionObjective] = NearestInsertion(
        m_optimizationMode, m_weightManager.W(), m_weightManager.StartPositions(),
        m_weightManager.EndPositions(), m_weightManager.Dependencies(), m_endTime);

    auto [twoOptedPaths, twoOptImprovement] = TwoOptPaths(
        m_optimizationMode, std::move(nearestInsertionPaths), m_weightManager.W(),
        m_weightManager.Dependencies(), m_endTime);

    m_bestResult.UpdateUpperBound(
        nearestInsertionObjective - twoOptImprovement,
        m_weightManager.TransformPathsBack(std::move(twoOptedPaths)));
}

double tsplp::MtspModel::ExploitFractionalSolution(const xt::xtensor<double, 3>& fractionalValues)
{
    auto exploitedPaths = tsplp::ExploitFractionalSolution(
        m_optimizationMode, fractionalValues, m_weightManager.W(), m_weightManager.StartPositions(),
        m_weightManager.EndPositions(), m_weightManager.Dependencies(), m_endTime);

    if (exploitedPaths.empty())
        return m_bestResult.GetBounds().Upper;

    auto [twoOptedPaths, _] = TwoOptPaths(
        m_optimizationMode, std::move(exploitedPaths), m_weightManager.W(),
        m_weightManager.Dependencies(), m_endTime);

    const auto exploitedObjective
        = CalculateObjective(m_optimizationMode, twoOptedPaths, m_weightManager.W());

    return m_bestResult
        .UpdateUpperBound(
            exploitedObjective, m_weightManager.TransformPathsBack(std::move(twoOptedPaths)))
        .Upper;
}

tsplp::LinearObjective tsplp::CreateObjective(
    xt::xarray<double> weights, xt::xarray<Variable> variables, std::optional<Variable> maxVariable)
{
    if (maxVariable)
        return CreateMaxObjective(weights, variables, *maxVariable);
    return CreateSumObjective(weights, variables);
}

tsplp::LinearObjective tsplp::CreateSumObjective(
    xt::xarray<double> weights, xt::xarray<Variable> variables)
{
    const auto A = variables.shape(0);
    const auto N = variables.shape(1);

    LinearObjective objective {};
    for (size_t a = 0; a < A; ++a)
    {
        for (size_t u = 0; u < N; ++u)
        {
            for (size_t v = 0; v < N; ++v)
                objective.Objective += weights(u, v) * variables(a, u, v);
        }
    }
    return objective;
}

tsplp::LinearObjective tsplp::CreateMaxObjective(
    xt::xarray<double> weights, xt::xarray<Variable> variables, Variable maxVariable)
{
    const auto A = variables.shape(0);
    const auto N = variables.shape(1);

    LinearObjective objective { .Objective = maxVariable, .AdditionalConstraints {} };
    objective.AdditionalConstraints.reserve(A);

    for (size_t a = 0; a < A; ++a)
    {
        LinearVariableComposition agentSum;
        for (size_t u = 0; u < N; ++u)
        {
            for (size_t v = 0; v < N; ++v)
                agentSum += weights(u, v) * variables(a, u, v);
        }
        objective.AdditionalConstraints.push_back(maxVariable >= std ::move(agentSum));
    }

    return objective;
}
