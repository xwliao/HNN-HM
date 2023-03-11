#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "wrapBCAGMMatching.h"
#include "wrapBCAGM_QUADMatching.h"
#include "wrapBCAGM3Matching.h"
#include "wrapBCAGM3_QUADMatching.h"

namespace py = pybind11;


PYBIND11_MODULE(BCAGMCore, m) {
    m.doc() = "Code for Block Coordinate Ascent Graph Matching BCAGM (T-PAMI 2017)"; // module docstring
    m.def("wrapBCAGMMatching", &wrapBCAGMMatching, "");
    m.def("wrapBCAGM_QUADMatching", &wrapBCAGM_QUADMatching, "");
    m.def("wrapBCAGM3Matching", &wrapBCAGM3Matching, "");
    m.def("wrapBCAGM3_QUADMatching", &wrapBCAGM3_QUADMatching, "");
}
