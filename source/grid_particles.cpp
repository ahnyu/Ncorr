// source/grid_particles.cpp
#include "grid_particles.h"
#include <omp.h>
#include <cstddef>   // For std::size_t
#include <vector>    // For std::vector

void gridParticles(
    const std::vector<double4>& particles,
    std::vector<std::vector<double4>>& particle_cells,
    const double3& cellSize,
    const int3& numCells
) {
    size_t num_cells = static_cast<size_t>(numCells.x) * numCells.y * numCells.z;
    int num_threads = omp_get_max_threads();
    size_t num_particles = particles.size();

    // Compute per-thread particle ranges
    std::vector<size_t> thread_starts(num_threads + 1, 0);

    size_t particles_per_thread = (num_particles + num_threads - 1) / num_threads;
    for (int t = 0; t < num_threads; ++t) {
        thread_starts[t] = t * particles_per_thread;
    }
    thread_starts[num_threads] = num_particles;

    // Initialize per-thread counts
    std::vector<std::vector<size_t>> thread_cell_counts(num_threads, std::vector<size_t>(num_cells, 0));

    // First pass: count particles per cell per thread
    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        size_t start = thread_starts[thread_id];
        size_t end = thread_starts[thread_id + 1];

        auto& cell_counts = thread_cell_counts[thread_id];

        for (size_t i = start; i < end; ++i) {
            const double4& p = particles[i];
            int x_cell = static_cast<int>(p.x / cellSize.x);
            if (x_cell == numCells.x) x_cell--;
            int y_cell = static_cast<int>(p.y / cellSize.y);
            if (y_cell == numCells.y) y_cell--;
            int z_cell = static_cast<int>(p.z / cellSize.z);
            if (z_cell == numCells.z) z_cell--;
            int cell_index = z_cell + numCells.z * (y_cell + numCells.y * x_cell);

            cell_counts[cell_index]++;
        }
    }

    // Compute total counts per cell
    std::vector<size_t> total_cell_counts(num_cells, 0);
    for (size_t cell = 0; cell < num_cells; ++cell) {
        size_t total_count = 0;
        for (int t = 0; t < num_threads; ++t) {
            total_count += thread_cell_counts[t][cell];
        }
        total_cell_counts[cell] = total_count;
    }

    // Compute cell start positions
    std::vector<size_t> cell_start_positions(num_cells + 1, 0);
    for (size_t cell = 0; cell < num_cells; ++cell) {
        cell_start_positions[cell + 1] = cell_start_positions[cell] + total_cell_counts[cell];
    }

    // Resize particle_cells and allocate exact sizes
    particle_cells.resize(num_cells);
    for (size_t cell = 0; cell < num_cells; ++cell) {
        particle_cells[cell].resize(total_cell_counts[cell]);
    }

    // Compute per-thread start positions within each cell
    std::vector<std::vector<size_t>> thread_cell_offsets(num_threads, std::vector<size_t>(num_cells, 0));
    for (size_t cell = 0; cell < num_cells; ++cell) {
        size_t offset = 0;
        for (int t = 0; t < num_threads; ++t) {
            thread_cell_offsets[t][cell] = offset;
            offset += thread_cell_counts[t][cell];
        }
    }

    // Second pass: write particles into particle_cells
    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        size_t start = thread_starts[thread_id];
        size_t end = thread_starts[thread_id + 1];

        auto& cell_offsets = thread_cell_offsets[thread_id];
        std::vector<size_t> local_offsets = cell_offsets;

        for (size_t i = start; i < end; ++i) {
            const double4& p = particles[i];
            int x_cell = static_cast<int>(p.x / cellSize.x);
            if (x_cell == numCells.x) x_cell--;
            int y_cell = static_cast<int>(p.y / cellSize.y);
            if (y_cell == numCells.y) y_cell--;
            int z_cell = static_cast<int>(p.z / cellSize.z);
            if (z_cell == numCells.z) z_cell--;

            int cell_index = z_cell + numCells.z * (y_cell + numCells.y * x_cell);

            size_t offset = local_offsets[cell_index];
            particle_cells[cell_index][offset] = p;
            local_offsets[cell_index]++;
        }
    }
}
