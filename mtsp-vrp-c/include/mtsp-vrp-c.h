#ifndef MTSP_VRP_C_H
#define MTSP_VRP_C_H

#include "mtsp-vrp-c_export.h"

#define MTSP_VRP_C_RESULT_SOLVED 0
#define MTSP_VRP_C_RESULT_TIMEOUT 1

#define MTSP_VRP_C_RESULT_INFEASIBLE -1
#define MTSP_VRP_C_RESULT_INVALID_INPUT_SIZE -2
#define MTSP_VRP_C_RESULT_INVALID_INPUT_POINTER -3

extern "C"
{
    MTSP_VRP_C_EXPORT int solve_mtsp_vrp(size_t numberOfAgents, size_t numberOfNodes, const int* start_positions, const int* end_positions, const double* weights,
        double* lowerBound, double* upperBound, int* paths, size_t* pathOffsets);
}

#endif