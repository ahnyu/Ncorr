// include/utils.h
#pragma once

#include "types.h"
#include <vector>

std::vector<int3> getShifts();

int getCellIndex(const double4& p, const double3& cellSize, const int3& numCells);

int4 initCellIndexBox(const double4& particle, const double3& cellSize, const int3& numCells);

int4 shiftCellIndexBox(int4 p_cell, int i, double3& rShift, const std::vector<int3>& shifts, const double3& box, const int3& numCells);

double sqrSep(const double4& p1, const double4& p2);

int getTriangleIndexThreeD(
    double sqrR1,
    double sqrR2,
    double sqrR3,
    const std::vector<double>& sqrBins,
    int numBins
);
