#ifndef WRAP_BCAGM_MATCHING_H
#define WRAP_BCAGM_MATCHING_H

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;


py::tuple wrapBCAGMMatching(
        py::array_t<int, py::array::f_style | py::array::forcecast> indH1, py::array_t<double, py::array::f_style | py::array::forcecast> valH1,
        py::array_t<int, py::array::f_style | py::array::forcecast> indH2, py::array_t<double, py::array::f_style | py::array::forcecast> valH2,
        py::array_t<int, py::array::f_style | py::array::forcecast> indH3, py::array_t<double, py::array::f_style | py::array::forcecast> valH3,
        int N1, int N2,
        py::array_t<double, py::array::f_style | py::array::forcecast> X,
        int maxIter, int adapt);

#endif
