#pragma once

#include "LinearVariableComposition.hpp"
#include "Model.hpp"
#include "Variable.hpp"
#include "WeightManager.hpp"

#include <xtensor/xtensor.hpp>

#include <chrono>
#include <functional>
#include <limits>
#include <mutex>
#include <optional>
#include <vector>

namespace tsplp
{
struct MtspResult
{
    std::vector<std::vector<size_t>> Paths {};
    double LowerBound = -std::numeric_limits<double>::max();
    double UpperBound = std::numeric_limits<double>::max();
    bool IsTimeoutHit = false;
};

class MtspModel
{
private:
    std::chrono::steady_clock::time_point m_startTime = std::chrono::steady_clock::now();
    std::chrono::steady_clock::time_point m_endTime;

    WeightManager m_weightManager;

    size_t A;
    size_t N;

    Model m_model;
    xt::xtensor<Variable, 3> X;

    LinearVariableComposition m_objective;

    MtspResult m_bestResult {};
    std::mutex m_bestResultMutex;

public:
    MtspModel(
        xt::xtensor<size_t, 1> startPositions, xt::xtensor<size_t, 1> endPositions,
        xt::xtensor<int, 2> weights, std::chrono::milliseconds timeout);

public:
    MtspResult BranchAndCutSolve(
        std::optional<size_t> noOfThreads = std::nullopt,
        std::function<void(const xt::xtensor<double, 3>&)> fractionalCallback = nullptr);

private:
    [[nodiscard]] std::vector<std::vector<size_t>> CreatePathsFromVariables(
        const Model& model) const;

    [[nodiscard]] std::vector<Variable> CalculateRecursivelyFixableVariables(Variable var) const;
};
}
