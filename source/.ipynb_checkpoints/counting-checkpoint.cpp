// source/counting.cpp
#include "counting.h"
#include "utils.h"
#include <cmath>
#include <omp.h>

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
) {
    const int numShifts = shifts.size();
    const size_t numTriangles = triangle_counts.size();

    // Use thread-local storage to avoid race conditions
    int num_threads = omp_get_max_threads();
    std::vector<std::vector<double>> local_triangle_counts(num_threads, std::vector<double>(numTriangles, 0.0));

    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        auto& counts = local_triangle_counts[thread_id];

        #pragma omp for schedule(dynamic)
        for (size_t idx1 = 0; idx1 < p1.size(); ++idx1) {
            const double4& particle1 = p1[idx1];
            int4 p1_cell = initCellIndexBox(particle1, cellSize, numCells);
            for (int nci1 = 0; nci1 < numShifts; ++nci1) {
                double3 rShift2;
                int4 p2_cell = shiftCellIndexBox(p1_cell, nci1, rShift2, shifts, box, numCells);
                int cell_counts2 = p2CellSize[p2_cell.w];

                const auto& p2_particles = p2Cell[p2_cell.w];

                for (int idx2 = 0; idx2 < cell_counts2; ++idx2) {
                    double4 particle2 = p2_particles[idx2];
                    particle2.x += rShift2.x;
                    particle2.y += rShift2.y;
                    particle2.z += rShift2.z;

                    double dx12 = particle1.x - particle2.x;
                    double dy12 = particle1.y - particle2.y;
                    double dz12 = particle1.z - particle2.z;
                    double sqrR1 = dx12 * dx12 + dy12 * dy12 + dz12 * dz12;

                    if (sqrR1 < sqrRMax && sqrR1 > sqrRMin) {
                        for (int nci2 = 0; nci2 < numShifts; ++nci2) {
                            double3 rShift3;
                            int4 p3_cell = shiftCellIndexBox(p1_cell, nci2, rShift3, shifts, box, numCells);
                            int cell_counts3 = p3CellSize[p3_cell.w];

                            const auto& p3_particles = p3Cell[p3_cell.w];

                            for (int idx3 = 0; idx3 < cell_counts3; ++idx3) {
                                double4 particle3 = p3_particles[idx3];
                                particle3.x += rShift3.x;
                                particle3.y += rShift3.y;
                                particle3.z += rShift3.z;

                                double dx13 = particle1.x - particle3.x;
                                double dy13 = particle1.y - particle3.y;
                                double dz13 = particle1.z - particle3.z;
                                double sqrR2 = dx13 * dx13 + dy13 * dy13 + dz13 * dz13;

                                double dx23 = particle2.x - particle3.x;
                                double dy23 = particle2.y - particle3.y;
                                double dz23 = particle2.z - particle3.z;
                                double sqrR3 = dx23 * dx23 + dy23 * dy23 + dz23 * dz23;

                                if (sqrR2 < sqrRMax && sqrR3 < sqrRMax && sqrR2 > sqrRMin && sqrR3 > sqrRMin) {
                                    int triangle_index = getTriangleIndexThreeD(sqrR1, sqrR2, sqrR3, sqrBins, numBins);

                                    if (triangle_index != -1) {
                                        double weight_product = particle1.w * particle2.w * particle3.w;
                                        counts[triangle_index] += weight_product;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // Combine counts from all threads
    for (int t = 0; t < num_threads; ++t) {
        for (size_t i = 0; i < numTriangles; ++i) {
            triangle_counts[i] += local_triangle_counts[t][i];
        }
    }
}
