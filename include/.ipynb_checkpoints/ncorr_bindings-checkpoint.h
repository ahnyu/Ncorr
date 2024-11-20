// include/ncorr_bindings.h
#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

void count_triangles(
    pybind11::array_t<double> p1_array,
    pybind11::array_t<double> p2_array,
    pybind11::array_t<double> p3_array,
    pybind11::array_t<double> bins_array,
    double box_x,
    double box_y,
    double box_z,
    int bin_type,
    pybind11::array_t<double> triangle_counts_output
);
