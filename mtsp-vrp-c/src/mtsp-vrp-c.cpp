#include "mtsp-vrp-c.h"

#include <MtspModel.hpp>
#include <TsplpExceptions.hpp>

#include <xtensor/xadapt.hpp>

#include <array>
#include <chrono>
#include <functional>

int solve_mtsp_vrp(
    size_t numberOfAgents, size_t numberOfNodes, const size_t* start_positions,
    const size_t* end_positions, const int* weights, int timeout_ms, size_t numberOfThreads,
    double* lowerBound, double* upperBound, size_t* paths, size_t* pathOffsets,
    int (*fractional_callback)(const double*))
{
    const auto startTime = std::chrono::steady_clock::now();

    if (numberOfAgents == 0 || numberOfNodes < 2 || numberOfAgents * 2 > numberOfNodes)
        return MTSP_VRP_C_NO_RESULT_INVALID_INPUT_SIZE;

    if (start_positions == nullptr || end_positions == nullptr || weights == nullptr
        || lowerBound == nullptr || upperBound == nullptr || paths == nullptr
        || pathOffsets == nullptr)
        return MTSP_VRP_C_NO_RESULT_INVALID_INPUT_POINTER;

    const std::array positionsShape = { numberOfAgents };
    const auto startPositions
        = xt::adapt(start_positions, numberOfAgents, xt::no_ownership {}, positionsShape);
    const auto endPositions
        = xt::adapt(end_positions, numberOfAgents, xt::no_ownership {}, positionsShape);

    const std::array weightsShape = { numberOfNodes, numberOfNodes };
    const auto weights_
        = xt::adapt(weights, numberOfNodes * numberOfNodes, xt::no_ownership {}, weightsShape);

    try
    {
        const auto timeout = std::chrono::milliseconds { timeout_ms }
            - std::chrono::duration_cast<std::chrono::milliseconds>(
                                 std::chrono::steady_clock::now() - startTime);

        tsplp::MtspModel model(startPositions, endPositions, weights_, timeout);

        const std::function<void(const xt::xtensor<double, 3>&)> callback = fractional_callback != nullptr
            ? [=](const xt::xtensor<double, 3>& tensor) {
                assert(tensor.shape() == (std::array{numberOfAgents, numberOfNodes, numberOfNodes}));
                fractional_callback(tensor.data()); }
            : std::function<void(const xt::xtensor<double, 3>&)> {};
        const auto result = model.BranchAndCutSolve(numberOfThreads, callback);

        *lowerBound = result.LowerBound;
        *upperBound = result.UpperBound;

        if (!result.IsTimeoutHit && result.UpperBound == std::numeric_limits<double>::max())
            return MTSP_VRP_C_NO_RESULT_INFEASIBLE;

        if (result.IsTimeoutHit && result.UpperBound == std::numeric_limits<double>::max())
            return MTSP_VRP_C_NO_RESULT_TIMEOUT;

        size_t offset = 0;
        for (size_t a = 0; a < numberOfAgents; ++a)
        {
            pathOffsets[a] = offset; // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
            auto length = result.Paths[a].size();
            if (startPositions[a] == endPositions[a])
                --length; // don't copy unneeded (duplicate) last entry
            // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
            std::copy_n(result.Paths[a].begin(), length, paths + offset);
            offset += length;
        }

        if (result.LowerBound >= result.UpperBound)
            return MTSP_VRP_C_RESULT_SOLVED;

        assert(result.IsTimeoutHit);
        return MTSP_VRP_C_RESULT_TIMEOUT;
    }
    catch (const tsplp::CyclicDependenciesException&)
    {
        return MTSP_VRP_C_CYCLIC_DEPENDENCIES;
    }
    catch (const tsplp::IncompatibleDependenciesException&)
    {
        return MTSP_VRP_C_INCOMPATIBLE_DEPENDENCIES;
    }
    catch (...)
    {
        return INT_MIN;
    }
}
