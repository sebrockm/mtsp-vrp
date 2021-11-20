#include "mtsp-vrp-c.h"
#include <MtspModel.hpp>

#include <array>
#include <xtensor/xadapt.hpp>

int solve_mtsp_vrp(size_t numberOfAgents, size_t numberOfNodes, const int* start_positions, const int* end_positions, const double* weights,
    double* lowerBound, double* upperBound, int* paths, size_t* pathOffsets)
{
    if (numberOfAgents == 0 || numberOfNodes < 2 || numberOfAgents * 2 > numberOfNodes)
        return -2;

    if (start_positions == nullptr || end_positions == nullptr || weights == nullptr ||
        lowerBound == nullptr || upperBound == nullptr || paths == nullptr || pathOffsets == nullptr)
        return -1;


    const std::vector<size_t> positionsShape = { numberOfAgents };
    const auto startPositions = xt::adapt(start_positions, numberOfAgents, xt::no_ownership{}, positionsShape);
    const auto endPositions = xt::adapt(end_positions, numberOfAgents, xt::no_ownership{}, positionsShape);

    const std::vector<size_t> weightsShape = { numberOfNodes, numberOfNodes };
    const auto weights_ = xt::adapt(weights, numberOfNodes * numberOfNodes, xt::no_ownership{}, weightsShape);

    tsplp::MtspModel model(startPositions, endPositions, weights_);
    auto const result = model.BranchAndCutSolve();

    *lowerBound = result.lowerBound;
    *upperBound = result.upperBound;

    if (result.lowerBound == -std::numeric_limits<double>::max() || result.upperBound == std::numeric_limits<double>::max())
        return 1;

    size_t offset = 0;
    for (size_t a = 0; a < numberOfAgents; ++a)
    {
        pathOffsets[a] = offset;
        std::copy(result.Paths[a].begin(), result.Paths[a].end(), paths + offset);
        offset += result.Paths[a].size();
    }

    return 0;
}