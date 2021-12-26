#pragma once

#include "LinearVariableComposition.hpp"
#include "Model.hpp"
#include "Variable.hpp"
#include "WeightManager.hpp"

#include <chrono>
#include <limits>
#include <vector>
#include <xtensor/xtensor.hpp>

namespace tsplp
{
    struct MtspResult
    {
        std::vector<std::vector<int>> Paths{};
        double LowerBound = -std::numeric_limits<double>::max();
        double UpperBound = std::numeric_limits<double>::max();
        bool IsTimeoutHit = false;
    };

    class MtspModel
    {
    private:
        WeightManager m_weightsManager;

        size_t A;
        size_t N;

        Model m_model;
        xt::xtensor<Variable, 3> X;

        LinearVariableComposition m_objective;

    public:
        MtspModel(xt::xtensor<int, 1> startPositions, xt::xtensor<int, 1> endPositions, xt::xtensor<int, 2> weights);

    public:
        MtspResult BranchAndCutSolve(std::chrono::milliseconds timeout);

    private:
        std::vector<std::vector<int>> CreatePathsFromVariables() const;
    };
}