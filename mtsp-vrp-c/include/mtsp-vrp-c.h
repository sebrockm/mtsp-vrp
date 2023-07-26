#ifndef MTSP_VRP_C_H
#define MTSP_VRP_C_H

#ifdef __cplusplus
extern "C"
{
#endif

#include "mtsp-vrp-c_export.h"

#ifdef __cplusplus
#include <cstddef>
#else
#include <stddef.h>
#endif

#define MTSP_VRP_C_RESULT_SOLVED 0
#define MTSP_VRP_C_RESULT_TIMEOUT 1

#define MTSP_VRP_C_OPTIMIZATION_MODE_SUM 0
#define MTSP_VRP_C_OPTIMIZATION_MODE_MAX 1

#define MTSP_VRP_C_NO_RESULT_TIMEOUT -1
#define MTSP_VRP_C_NO_RESULT_INFEASIBLE -2
#define MTSP_VRP_C_NO_RESULT_INVALID_INPUT_SIZE -3
#define MTSP_VRP_C_NO_RESULT_INVALID_INPUT_POINTER -4
#define MTSP_VRP_C_CYCLIC_DEPENDENCIES -5
#define MTSP_VRP_C_INCOMPATIBLE_DEPENDENCIES -6
#define MTSP_VRP_C_INVALID_OPTIMIZATION_MODE -7

    MTSP_VRP_C_EXPORT int solve_mtsp_vrp(
        size_t numberOfAgents, size_t numberOfNodes, const size_t* start_positions,
        const size_t* end_positions, const int* weights, int optimizationMode, int timeout_ms,
        size_t numberOfThreads, double* lowerBound, double* upperBound, size_t* paths,
        size_t* pathOffsets, int (*fractional_callback)(const double*));

#ifdef __cplusplus
}
#endif

#endif
