#pragma once

#include "Model.hpp"
#include "Variable.hpp"

#include <vector>
#include <xtensor/xtensor.hpp>

namespace tsplp
{
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

    public:
        MtspModel(std::vector<int> startPositions, std::vector<int> endPositions, std::vector<double> weights);
    };
}