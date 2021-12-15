#pragma once

#include "LinearVariableComposition.hpp"
#include "Model.hpp"
#include "Variable.hpp"

#include <chrono>
#include <limits>
#include <optional>
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
        xt::xtensor<int, 1> m_startPositions;
        xt::xtensor<int, 1> m_endPositions;

        size_t A;
        size_t N;

        Model m_model;

        xt::xtensor<double, 2> W;
        xt::xtensor<Variable, 3> X;

        LinearVariableComposition m_objective;

    public:
        MtspModel(xt::xtensor<int, 1> startPositions, xt::xtensor<int, 1> endPositions, xt::xtensor<int, 2> weights);

    public:
        MtspResult BranchAndCutSolve(std::chrono::milliseconds timeout,
            std::optional<int> heuristicObjective = {}, std::optional<std::vector<std::vector<int>>> heuristicPaths = {});

    private:
        std::vector<std::vector<int>> CreatePathsFromVariables() const;
    };
}