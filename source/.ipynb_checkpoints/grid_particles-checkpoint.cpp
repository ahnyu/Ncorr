// source/grid_particles.cpp
#include "grid_particles.h"
#include <omp.h>
#include <cstddef>   // For std::size_t
#include <vector>    // For std::vector

using std::size_t;

void gridParticles(
    const std::vector<double4>& particles,
    std::vector<std::vector<double4>>& particle_cells,
    const double3& cellSize,
    const int3& numCells
) {
    size_t num_cells = static_cast<size_t>(numCells.x) * numCells.y * numCells.z;
    particle_cells.resize(num_cells);

    // Initialize particle_cells with empty vectors
    #pragma omp parallel for
    for (size_t i = 0; i < num_cells; ++i) {
        particle_cells[i].clear();
    }

    // Each thread will have its own local copy to avoid locks
    int num_threads = omp_get_max_threads();
    std::vector<std::vector<std::vector<double4>>> local_cells(num_threads, std::vector<std::vector<double4>>(num_cells));

    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        auto& thread_cells = local_cells[thread_id];

        #pragma omp for nowait
        for (size_t i = 0; i < particles.size(); ++i) {
            const double4& p = particles[i];
            int x_cell = static_cast<int>(p.x / cellSize.x);
            if (x_cell == numCells.x) x_cell--;
            int y_cell = static_cast<int>(p.y / cellSize.y);
            if (y_cell == numCells.y) y_cell--;
            int z_cell = static_cast<int>(p.z / cellSize.z);
            if (z_cell == numCells.z) z_cell--;
            int cell_index = z_cell + numCells.z * (y_cell + numCells.y * x_cell);

            thread_cells[cell_index].push_back(p);
        }
    }

    // Merge local cells into global cells
    #pragma omp parallel for
    for (size_t i = 0; i < num_cells; ++i) {
        for (int t = 0; t < num_threads; ++t) {
            particle_cells[i].insert(particle_cells[i].end(), local_cells[t][i].begin(), local_cells[t][i].end());
        }
    }
}
