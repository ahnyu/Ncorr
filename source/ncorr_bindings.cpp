// source/ncorr_bindings.cpp
#include "ncorr_bindings.h"
#include "types.h"
#include "utils.h"
#include "grid_particles.h"
#include "counting.h"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <omp.h>
#include <iostream>
#include <chrono>

namespace py = pybind11;

void count_triangles(
    py::array_t<double> p1_array,
    py::array_t<double> p2_array,
    py::array_t<double> p3_array,
    py::array_t<double> bins_array,
    double box_x,
    double box_y,
    double box_z,
    int bin_type,
    py::array_t<double> triangle_counts_output
) {
    auto start = std::chrono::high_resolution_clock::now();
    // Check input dimensions and get data pointers
    if (p1_array.ndim() != 2 || p1_array.shape(1) != 4)
        throw std::runtime_error("p1_array must have shape (N, 4)");
    if (p2_array.ndim() != 2 || p2_array.shape(1) != 4)
        throw std::runtime_error("p2_array must have shape (N, 4)");
    if (p3_array.ndim() != 2 || p3_array.shape(1) != 4)
        throw std::runtime_error("p3_array must have shape (N, 4)");
    if (bins_array.ndim() != 1)
        throw std::runtime_error("bins_array must be a 1D array");

    auto p1_buf = p1_array.request();
    auto p2_buf = p2_array.request();
    auto p3_buf = p3_array.request();
    auto bins_buf = bins_array.request();
    auto triangle_counts_buf = triangle_counts_output.request();

    size_t Np1 = p1_buf.shape[0];
    size_t Np2 = p2_buf.shape[0];
    size_t Np3 = p3_buf.shape[0];
    size_t num_bins = bins_buf.shape[0] - 1;
    pybind11::ssize_t num_triangles = num_bins * num_bins * num_bins;

    if (triangle_counts_buf.size != num_triangles)
        throw std::runtime_error("triangle_counts_output has incorrect size");

    double* p1_ptr = static_cast<double*>(p1_buf.ptr);
    double* p2_ptr = static_cast<double*>(p2_buf.ptr);
    double* p3_ptr = static_cast<double*>(p3_buf.ptr);
    double* bins_ptr = static_cast<double*>(bins_buf.ptr);
    double* counts_ptr = static_cast<double*>(triangle_counts_buf.ptr);

    // Convert inputs to C++ data structures
    std::vector<double4> p1(Np1), p2(Np2), p3(Np3);
    std::vector<double> bins(bins_ptr, bins_ptr + bins_buf.shape[0]);
    auto check1 = std::chrono::high_resolution_clock::now();

    #pragma omp parallel for
    for (size_t i = 0; i < Np1; ++i) {
        p1[i] = {p1_ptr[4 * i], p1_ptr[4 * i + 1], p1_ptr[4 * i + 2], p1_ptr[4 * i + 3]};
    }
    #pragma omp parallel for
    for (size_t i = 0; i < Np2; ++i) {
        p2[i] = {p2_ptr[4 * i], p2_ptr[4 * i + 1], p2_ptr[4 * i + 2], p2_ptr[4 * i + 3]};
    }
    #pragma omp parallel for
    for (size_t i = 0; i < Np3; ++i) {
        p3[i] = {p3_ptr[4 * i], p3_ptr[4 * i + 1], p3_ptr[4 * i + 2], p3_ptr[4 * i + 3]};
    }
    auto check2 = std::chrono::high_resolution_clock::now();
    
    double3 box = {box_x, box_y, box_z};
    double rMax = bins.back();
    double rMin = bins.front();

    double3 cellSize = {rMax, rMax, rMax};
    int3 numCells = {
        static_cast<int>(std::ceil(box.x / cellSize.x)),
        static_cast<int>(std::ceil(box.y / cellSize.y)),
        static_cast<int>(std::ceil(box.z / cellSize.z))
    };

    std::vector<int3> shifts = getShifts();

    // Prepare bins
    std::vector<double> sqrBins(bins.size());
    for (size_t i = 0; i < bins.size(); ++i) {
        sqrBins[i] = bins[i] * bins[i];
    }

    // Initialize triangle counts
    std::vector<double> triangle_counts(num_triangles, 0.0);

    // Grid particles
    std::vector<std::vector<double4>> p2Cell, p3Cell;
    gridParticles(p2, p2Cell, cellSize, numCells);
    gridParticles(p3, p3Cell, cellSize, numCells);

    std::vector<int> p2CellSize(p2Cell.size()), p3CellSize(p3Cell.size());
    for (size_t i = 0; i < p2Cell.size(); ++i) {
        p2CellSize[i] = p2Cell[i].size();
    }
    for (size_t i = 0; i < p3Cell.size(); ++i) {
        p3CellSize[i] = p3Cell[i].size();
    }
    auto check3 = std::chrono::high_resolution_clock::now();
    
    // Call the counting function
    countTrianglesThreeDBoxCPU(
        p1,
        p2Cell,
        p3Cell,
        p2CellSize,
        p3CellSize,
        sqrBins,
        num_bins,
        triangle_counts,
        rMax * rMax,
        rMin * rMin,
        cellSize,
        box,
        numCells,
        shifts
    );
    auto check4 = std::chrono::high_resolution_clock::now();
    
    auto duration1 = std::chrono::duration_cast<std::chrono::microseconds>(check1 - start);
    auto duration2 = std::chrono::duration_cast<std::chrono::microseconds>(check2 - check1);
    auto duration3 = std::chrono::duration_cast<std::chrono::microseconds>(check3 - check2);
    auto duration4 = std::chrono::duration_cast<std::chrono::microseconds>(check4 - check3);
    
    std::cout << "Phase 1 time cost: " << duration1.count() << " microseconds" << std::endl;
    std::cout << "Phase 2 time cost: " << duration2.count() << " microseconds" << std::endl;
    std::cout << "Phase 3 time cost: " << duration3.count() << " microseconds" << std::endl;
    std::cout << "Phase 3 time cost: " << duration3.count() << " microseconds" << std::endl;
    

    // Copy results to output array
    std::copy(triangle_counts.begin(), triangle_counts.end(), counts_ptr);
}

PYBIND11_MODULE(ncorr_module, m) {
    m.def("count_triangles", &count_triangles, "Count triangles in 3D box",
          py::arg("p1_array"),
          py::arg("p2_array"),
          py::arg("p3_array"),
          py::arg("bins_array"),
          py::arg("box_x"),
          py::arg("box_y"),
          py::arg("box_z"),
          py::arg("bin_type"),
          py::arg("triangle_counts_output"));
}
