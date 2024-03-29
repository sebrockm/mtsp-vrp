#pragma once

#include "LinearConstraint.hpp"
#include "LinearVariableComposition.hpp"
#include "Model.hpp"
#include "MtspResult.hpp"
#include "Variable.hpp"
#include "WeightManager.hpp"

#include <xtensor/xtensor.hpp>

#include <chrono>
#include <functional>
#include <optional>
#include <vector>

namespace tsplp
{
enum class OptimizationMode
{
    Sum,
    Max
};

struct LinearObjective
{
    LinearVariableComposition Objective;
    std::vector<LinearConstraint> AdditionalConstraints;
};

class MtspModel
{
private:
    std::chrono::steady_clock::time_point m_startTime = std::chrono::steady_clock::now();
    std::chrono::steady_clock::time_point m_endTime;

    WeightManager m_weightManager;

    OptimizationMode m_optimizationMode;

    size_t A;
    size_t N;

    Model m_model;
    xt::xtensor<Variable, 3> X;

    LinearObjective m_objective;

    MtspResult m_bestResult {};

    std::string m_name;

public:
    MtspModel(
        xt::xtensor<size_t, 1> startPositions, xt::xtensor<size_t, 1> endPositions,
        xt::xtensor<double, 2> weights, OptimizationMode optimizationMode,
        std::chrono::milliseconds timeout, std::string name = "Model");

public:
    void BranchAndCutSolve(
        std::optional<size_t> noOfThreads = std::nullopt,
        std::function<void(const xt::xtensor<double, 3>&)> fractionalCallback = nullptr);

    [[nodiscard]] const MtspResult& GetResult() const { return m_bestResult; }

private:
    void CreateInitialResult();
    double ExploitFractionalSolution(const xt::xtensor<double, 3>& fractionalValues);

    [[nodiscard]] std::vector<std::vector<size_t>> CreatePathsFromVariables(
        const Model& model) const;

    [[nodiscard]] std::vector<Variable> CalculateRecursivelyFixableVariables(Variable var) const;
};

[[nodiscard]] LinearObjective CreateObjective(
    xt::xarray<double> weights, xt::xarray<Variable> variables,
    std::optional<Variable> maxVariable);

[[nodiscard]] LinearObjective CreateSumObjective(
    xt::xarray<double> weights, xt::xarray<Variable> variables);

[[nodiscard]] LinearObjective CreateMaxObjective(
    xt::xarray<double> weights, xt::xarray<Variable> variables, Variable maxVariable);
}
