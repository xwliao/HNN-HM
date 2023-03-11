#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "ComputeFeature.h"

namespace py = pybind11;


py::array_t<double, py::array::c_style> wrapComputeFeatureSimple(
        py::array_t<double, py::array::c_style | py::array::forcecast> P,
        int i, int j, int k)
{
    /*
     * INPUT:
     *   P: nP x 2 (row-major)
     *   i, j, k: row index of P
     *
     * OUTPUT:
     *   F: (3,) (row-major)
    */

    py::buffer_info Pbuf = P.request();

    if (Pbuf.ndim != 2 || Pbuf.shape[1] != 2)
        throw std::runtime_error("Shape of P must be (nP, 2)");

    double *pP = (double *) Pbuf.ptr;

    const int nFeature = 3;
    auto F = py::array_t<double, py::array::c_style>({nFeature});
    py::buffer_info Fbuf = F.request();
    double *pF = (double *) Fbuf.ptr;

    computeFeatureSimple(pP, i, j, k, pF);

    return F;
}


py::array_t<double, py::array::c_style> wrapComputeFeatureSimple2(
        py::array_t<double, py::array::c_style | py::array::forcecast> P,
        py::array_t<int, py::array::c_style | py::array::forcecast> T)
{
    /*
     * INPUT:
     *   P: nP x 2 (row-major)
     *   T: nT x 3 (row-major)
     *
     * OUTPUT:
     *   F: nT x 3 (row-major)
    */

    py::buffer_info Pbuf = P.request();
    py::buffer_info Tbuf = T.request();

    if (Pbuf.ndim != 2 || Pbuf.shape[1] != 2)
        throw std::runtime_error("Shape of P must be (nP, 2)");

    if (Tbuf.ndim != 2 || Tbuf.shape[1] != 3)
        throw std::runtime_error("Shape of T must be (nT, 3)");

    int nP = Pbuf.shape[0];
    int nT = Tbuf.shape[0];

    double *pP = (double *) Pbuf.ptr;
    int *pT = (int *) Tbuf.ptr;

    const int nFeature = 3;
    auto F = py::array_t<double, py::array::c_style>({nT, nFeature});
    py::buffer_info Fbuf = F.request();
    double *pF = (double *) Fbuf.ptr;

    computeFeatureSimple2(pP, nP, pT, nT, pF);

    return F;
}


py::tuple wrapComputeFeature(
        py::array_t<double, py::array::c_style | py::array::forcecast> P1,
        py::array_t<double, py::array::c_style | py::array::forcecast> P2,
        py::array_t<int, py::array::c_style | py::array::forcecast> T1)
{
    /*
     * INPUT:
     *   P1: nP1 x 2 (row-major)
     *   P2: nP2 x 2 (row-major)
     *   T1: nT1 x 3 (row-major)
     *
     * OUTPUT:
     *   F1: nT1 x 3 (row-major)
     *   F2: (nP2*nP2*nP2) x 3 (row-major)
    */

    py::buffer_info P1buf = P1.request();
    py::buffer_info P2buf = P2.request();
    py::buffer_info T1buf = T1.request();

    if (P1buf.ndim != 2 || P1buf.shape[1] != 2)
        throw std::runtime_error("Shape of P1 must be (nP1, 2)");

    if (P2buf.ndim != 2 || P2buf.shape[1] != 2)
        throw std::runtime_error("Shape of P2 must be (nP2, 2)");

    if (T1buf.ndim != 2 || T1buf.shape[1] != 3)
        throw std::runtime_error("Shape of T1 must be (nT1, 3)");

    int nP1 = P1buf.shape[0];
    int nP2 = P2buf.shape[0];
    int nT1 = T1buf.shape[0];

    double *pP1 = (double *) P1buf.ptr;
    double *pP2 = (double *) P2buf.ptr;
    int *pT1 = (int *) T1buf.ptr;

    int nF1 = nT1;
    int nF2 = nP2 * nP2 * nP2;
    const int nFeature = 3;

    auto F1 = py::array_t<double, py::array::c_style>({nF1, nFeature});
    py::buffer_info F1buf = F1.request();
    double *pF1 = (double *) F1buf.ptr;

    auto F2 = py::array_t<double, py::array::c_style>({nF2, nFeature});
    py::buffer_info F2buf = F2.request();
    double *pF2 = (double *) F2buf.ptr;

    computeFeature(pP1, nP1, pP2, nP2, pT1, nT1, pF1, pF2);

    return py::make_tuple(F1, F2);
}


PYBIND11_MODULE(ComputeFeatureCore, m) {
    m.doc() = "Functions to compute high-order feature."; // module docstring
    m.def("wrapComputeFeatureSimple", &wrapComputeFeatureSimple, "");
    m.def("wrapComputeFeatureSimple2", &wrapComputeFeatureSimple2, "");
    m.def("wrapComputeFeature", &wrapComputeFeature, "");
}
