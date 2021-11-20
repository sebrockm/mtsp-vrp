#ifndef MTSP_VRP_C_H
#define MTSP_VRP_C_H

extern "C"
{
    int solve_mtsp_vrp(size_t numberOfAgents, size_t numberOfNodes, const int* start_positions, const int* end_positions, const double* weights,
        double* lowerBound, double* upperBound, int* paths, size_t* pathOffsets);
}

#endif