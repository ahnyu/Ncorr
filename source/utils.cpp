// source/utils.cpp
#include "utils.h"
#include <cmath>

std::vector<int3> getShifts() {
    std::vector<int3> shifts;
    for (int i = -1; i <= 1; ++i) {
        for (int j = -1; j <= 1; ++j) {
            for (int k = -1; k <= 1; ++k) {
                shifts.push_back({i, j, k});
            }
        }
    }
    return shifts;
}

int getCellIndex(const double4& p, const double3& cellSize, const int3& numCells) {
    int x_cell = static_cast<int>(p.x / cellSize.x);
    if (x_cell == numCells.x) x_cell--;
    int y_cell = static_cast<int>(p.y / cellSize.y);
    if (y_cell == numCells.y) y_cell--;
    int z_cell = static_cast<int>(p.z / cellSize.z);
    if (z_cell == numCells.z) z_cell--;
    return z_cell + numCells.z * (y_cell + numCells.y * x_cell);
}

int4 initCellIndexBox(const double4& particle, const double3& cellSize, const int3& numCells) {
    int4 cellIndex = {
        int(particle.x / cellSize.x),
        int(particle.y / cellSize.y),
        int(particle.z / cellSize.z),
        0
    };

    if (cellIndex.x == numCells.x) cellIndex.x--;
    if (cellIndex.y == numCells.y) cellIndex.y--;
    if (cellIndex.z == numCells.z) cellIndex.z--;
    cellIndex.w = cellIndex.z + numCells.z * (cellIndex.y + numCells.y * cellIndex.x);
    return cellIndex;
}

int4 shiftCellIndexBox(int4 p_cell, int i, double3& rShift, const std::vector<int3>& shifts, const double3& box, const int3& numCells) {
    p_cell.x += shifts[i].x;
    p_cell.y += shifts[i].y;
    p_cell.z += shifts[i].z;
    rShift.x = 0.0;
    rShift.y = 0.0;
    rShift.z = 0.0;
    if (p_cell.x == numCells.x) {
        p_cell.x = 0;
        rShift.x = box.x;
    }
    if (p_cell.y == numCells.y) {
        p_cell.y = 0;
        rShift.y = box.y;
    }
    if (p_cell.z == numCells.z) {
        p_cell.z = 0;
        rShift.z = box.z;
    }
    if (p_cell.x == -1) {
        p_cell.x = numCells.x - 1;
        rShift.x = -box.x;
    }
    if (p_cell.y == -1) {
        p_cell.y = numCells.y - 1;
        rShift.y = -box.y;
    }
    if (p_cell.z == -1) {
        p_cell.z = numCells.z - 1;
        rShift.z = -box.z;
    }
    p_cell.w = p_cell.z + numCells.z * (p_cell.y + numCells.y * p_cell.x);
    return p_cell;
}

double sqrSep(const double4& p1, const double4& p2) {
    double dx = p1.x - p2.x;
    double dy = p1.y - p2.y;
    double dz = p1.z - p2.z;

    return dx * dx + dy * dy + dz * dz;
}

int getTriangleIndexThreeD(
    double sqrR1,
    double sqrR2,
    double sqrR3,
    const std::vector<double>& sqrBins,
    int numBins
) {
    // Sort the lengths in ascending order
    double temp;
    if (sqrR1 > sqrR2) {
        temp = sqrR1; sqrR1 = sqrR2; sqrR2 = temp;
    }
    if (sqrR1 > sqrR3) {
        temp = sqrR1; sqrR1 = sqrR3; sqrR3 = temp;
    }
    if (sqrR2 > sqrR3) {
        temp = sqrR2; sqrR2 = sqrR3; sqrR3 = temp;
    }
    // Check if the given lengths can form a triangle
    if (sqrt(sqrR1) + sqrt(sqrR2) <= sqrt(sqrR3)) {
        return -1; // The lengths cannot form a triangle
    }

    // Find the intervals for each length
    int idx1 = -1, idx2 = -1, idx3 = -1;
    for (int i = 0; i < numBins; ++i) {
        if (sqrR1 >= sqrBins[i] && sqrR1 < sqrBins[i + 1]) {
            idx1 = i;
        }
        if (sqrR2 >= sqrBins[i] && sqrR2 < sqrBins[i + 1]) {
            idx2 = i;
        }
        if (sqrR3 >= sqrBins[i] && sqrR3 < sqrBins[i + 1]) {
            idx3 = i;
        }
    }

    if (idx1 == -1 || idx2 == -1 || idx3 == -1) {
        return -1; // At least one of the lengths is not in the bin range
    }

    // Flatten the 3D index to 1D
    int index = idx3 + idx2 * numBins + idx1 * numBins * numBins;

    return index;
}
