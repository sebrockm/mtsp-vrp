#include "mtsp-vrp-c.h"
#include <MtspModel.hpp>

#include <array>
#include <chrono>
#include <xtensor/xadapt.hpp>

int solve_mtsp_vrp(size_t numberOfAgents, size_t numberOfNodes, const int* start_positions, const int* end_positions, const double* weights, int timeout,
    double* lowerBound, double* upperBound, int* paths, size_t* pathOffsets)
{
    if (numberOfAgents == 0 || numberOfNodes < 2 || numberOfAgents * 2 > numberOfNodes)
        return MTSP_VRP_C_NO_RESULT_INVALID_INPUT_SIZE;

    if (start_positions == nullptr || end_positions == nullptr || weights == nullptr ||
        lowerBound == nullptr || upperBound == nullptr || paths == nullptr || pathOffsets == nullptr)
        return MTSP_VRP_C_NO_RESULT_INVALID_INPUT_POINTER;

    const std::array positionsShape = { numberOfAgents };
    const auto startPositions = xt::adapt(start_positions, numberOfAgents, xt::no_ownership{}, positionsShape);
    const auto endPositions = xt::adapt(end_positions, numberOfAgents, xt::no_ownership{}, positionsShape);

    const std::array weightsShape = { numberOfNodes, numberOfNodes };
    const auto weights_ = xt::adapt(weights, numberOfNodes * numberOfNodes, xt::no_ownership{}, weightsShape);

    tsplp::MtspModel model(startPositions, endPositions, weights_);
    auto const result = model.BranchAndCutSolve(std::chrono::milliseconds{ timeout });

    *lowerBound = result.LowerBound;
    *upperBound = result.UpperBound;

    if (!result.IsTimeoutHit && result.UpperBound == std::numeric_limits<double>::max())
        return MTSP_VRP_C_NO_RESULT_INFEASIBLE;

    if (result.IsTimeoutHit && result.UpperBound == std::numeric_limits<double>::max())
        return MTSP_VRP_C_NO_RESULT_TIMEOUT;

    size_t offset = 0;
    for (size_t a = 0; a < numberOfAgents; ++a)
    {
        pathOffsets[a] = offset;
        std::copy(result.Paths[a].begin(), result.Paths[a].end(), paths + offset);
        offset += result.Paths[a].size();
    }

    if (result.LowerBound >= result.UpperBound)
        return MTSP_VRP_C_RESULT_SOLVED;

    assert(result.timeout);
    return MTSP_VRP_C_RESULT_TIMEOUT;
}