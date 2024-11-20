// include/counting.h
#pragma once

#include "types.h"
#include <vector>

void countTrianglesThreeDBoxCPU(
    const std::vector<double4>& p1,
    const std::vector<std::vector<double4>>& p2Cell,
    const std::vector<std::vector<double4>>& p3Cell,
    const std::vector<int>& p2CellSize,
    const std::vector<int>& p3CellSize,
    const std::vector<double>& sqrBins,
    int numBins,
    std::vector<double>& triangle_counts,
    double sqrRMax,
    double sqrRMin,
    const double3& cellSize,
    const double3& box,
    const int3& numCells,
    const std::vector<int3>& shifts
);
