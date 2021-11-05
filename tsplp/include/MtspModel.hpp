#pragma once

#include "LinearVariableComposition.hpp"
#include "Model.hpp"
#include "Variable.hpp"

#include <limits>
#include <vector>
#include <xtensor/xtensor.hpp>

namespace tsplp
{
    struct MtspResult
    {
        xt::xtensor<int, 2> Paths{};
        double lowerBound = -std::numeric_limits<double>::max();
        double upperBound = std::numeric_limits<double>::max();
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
        MtspModel(std::vector<int> startPositions, std::vector<int> endPositions, std::vector<double> weights);

    public:
        MtspResult BranchAndCutSolve();
    };
}