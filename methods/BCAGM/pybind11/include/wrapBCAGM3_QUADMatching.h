#ifndef WRAP_BCAGM3_QUAD_MATCHING_H
#define WRAP_BCAGM3_QUAD_MATCHING_H

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;


py::tuple wrapBCAGM3_QUADMatching(
        py::array_t<int, py::array::f_style | py::array::forcecast> indH1, py::array_t<double, py::array::f_style | py::array::forcecast> valH1,
        py::array_t<int, py::array::f_style | py::array::forcecast> indH2, py::array_t<double, py::array::f_style | py::array::forcecast> valH2,
        py::array_t<int, py::array::f_style | py::array::forcecast> indH3, py::array_t<double, py::array::f_style | py::array::forcecast> valH3,
        int N1, int N2,
        py::array_t<double, py::array::f_style | py::array::forcecast> X,
        int subroutine, int adapt);

#endif
