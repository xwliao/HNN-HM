#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "RRWHM.h"

namespace py = pybind11;


py::array_t<double, py::array::f_style> wrapRRWHM(
        py::array_t<double, py::array::f_style | py::array::forcecast> X,
        py::array_t<int, py::array::f_style | py::array::forcecast> indH1, py::array_t<double, py::array::f_style | py::array::forcecast> valH1,
        py::array_t<int, py::array::f_style | py::array::forcecast> indH2, py::array_t<double, py::array::f_style | py::array::forcecast> valH2,
        py::array_t<int, py::array::f_style | py::array::forcecast> indH3, py::array_t<double, py::array::f_style | py::array::forcecast> valH3,
        int nIter, double c)
{
    py::buffer_info Xbuf = X.request();

    py::buffer_info indH1buf = indH1.request();
    py::buffer_info valH1buf = valH1.request();

    py::buffer_info indH2buf = indH2.request();
    py::buffer_info valH2buf = valH2.request();

    py::buffer_info indH3buf = indH3.request();
    py::buffer_info valH3buf = valH3.request();

    if (Xbuf.ndim != 2)
        throw std::runtime_error("Number of dimensions of X must be two");

    if (indH1buf.ndim != 2 || indH1buf.shape[1] != 1)
        throw std::runtime_error("Shape of indH1 must be (N1, 1)");

    if (indH1buf.shape[0] != valH1buf.size)
        throw std::runtime_error("indH1.shape[0] must match valH1.size");

    if (indH2buf.ndim != 2 || indH2buf.shape[1] != 2)
        throw std::runtime_error("Shape of indH2 must be (N2, 2)");

    if (indH2buf.shape[0] != valH2buf.size)
        throw std::runtime_error("indH2.shape[0] must match valH2.size");

    if (indH3buf.ndim != 2 || indH3buf.shape[1] != 3)
        throw std::runtime_error("Shape of indH3 must be (N3, 3)");

    if (indH3buf.shape[0] != valH3buf.size)
        throw std::runtime_error("indH3.shape[0] must match valH3.size");

    int N2 = Xbuf.shape[0];
    int N1 = Xbuf.shape[1];

    int Nt1 = valH1buf.size;
    int Nt2 = valH2buf.size;
    int Nt3 = valH3buf.size;

    double *pX = (double *) Xbuf.ptr;

    int *pIndH1 = (int *) indH1buf.ptr;
    double *pValH1 = (double *) valH1buf.ptr;

    int *pIndH2 = (int *) indH2buf.ptr;
    double *pValH2 = (double *) valH2buf.ptr;

    int *pIndH3 = (int *) indH3buf.ptr;
    double *pValH3 = (double *) valH3buf.ptr;
    
    auto Xout = py::array_t<double, py::array::f_style>(Xbuf.shape);
    py::buffer_info Xoutbuf = Xout.request();
    double *pXout = (double *) Xoutbuf.ptr;

    // TODO: Rename function name from RRWM(original name in the author's code) to RRWHM
    // Note: the order that pass N2 and then N1 is important here,
    //       ohterwise it will return wrong answers for cases that N2 > N1.
    RRWM(pX, N2, N1,
         pIndH1, pValH1, Nt1,
         pIndH2, pValH2, Nt2,
         pIndH3, pValH3, Nt3,
         nIter, &c,
         pXout);

    return Xout;
}


PYBIND11_MODULE(RRWHMCore, m) {
    m.doc() = "Reweighted Random Walks Hyper-graph Matching (CVPR 2011)"; // module docstring
    m.def("wrapRRWHM", &wrapRRWHM, "Reweighted Random Walks Hyper-graph Matching (CVPR 2011)");
}
