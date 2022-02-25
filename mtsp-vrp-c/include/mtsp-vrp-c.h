#ifndef MTSP_VRP_C_H
#define MTSP_VRP_C_H

#include <stddef.h>

#include "mtsp-vrp-c_export.h"

#define MTSP_VRP_C_RESULT_SOLVED 0
#define MTSP_VRP_C_RESULT_TIMEOUT 1

#define MTSP_VRP_C_NO_RESULT_TIMEOUT -1
#define MTSP_VRP_C_NO_RESULT_INFEASIBLE -2
#define MTSP_VRP_C_NO_RESULT_INVALID_INPUT_SIZE -3
#define MTSP_VRP_C_NO_RESULT_INVALID_INPUT_POINTER -4
#define MTSP_VRP_C_CYCLIC_DEPENDENCIES -5
#define MTSP_VRP_C_INCOMPATIBLE_DEPENDENCIES -6

extern "C"
{
    MTSP_VRP_C_EXPORT int solve_mtsp_vrp(size_t numberOfAgents, size_t numberOfNodes, const size_t* start_positions, const size_t* end_positions, const int* weights, int timeout_ms,
        size_t numberOfThreads, double* lowerBound, double* upperBound, size_t* paths, size_t* pathOffsets);
}

#endif
