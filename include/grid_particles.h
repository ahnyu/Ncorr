// include/grid_particles.h
#pragma once

#include "types.h"
#include <vector>

void gridParticles(
    const std::vector<double4>& particles,
    std::vector<std::vector<double4>>& particle_cells,
    const double3& cellSize,
    const int3& numCells
);
