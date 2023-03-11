#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "BCAGM3_LinSolver.h"
#include "wrapBCAGM3Matching.h"

namespace py = pybind11;


py::tuple wrapBCAGM3Matching(
        py::array_t<int, py::array::f_style | py::array::forcecast> indH1, py::array_t<double, py::array::f_style | py::array::forcecast> valH1,
        py::array_t<int, py::array::f_style | py::array::forcecast> indH2, py::array_t<double, py::array::f_style | py::array::forcecast> valH2,
        py::array_t<int, py::array::f_style | py::array::forcecast> indH3, py::array_t<double, py::array::f_style | py::array::forcecast> valH3,
        int N1, int N2,
        py::array_t<double, py::array::f_style | py::array::forcecast> X,
        int maxIter, int adapt)
{
    py::buffer_info Xbuf = X.request();

    py::buffer_info indH1buf = indH1.request();
    py::buffer_info valH1buf = valH1.request();

    py::buffer_info indH2buf = indH2.request();
    py::buffer_info valH2buf = valH2.request();

    py::buffer_info indH3buf = indH3.request();
    py::buffer_info valH3buf = valH3.request();

    if (Xbuf.size != 3 * N1 * N2)
        throw std::runtime_error("X.size must be 3 * N1 * N2");

    if (indH1buf.ndim != 2 || indH1buf.shape[0] != 1)
        throw std::runtime_error("Shape of indH1 must be (1, N1)");

    if (indH1buf.shape[1] != valH1buf.size)
        throw std::runtime_error("indH1.shape[1] must match valH1.size");

    if (indH2buf.ndim != 2 || indH2buf.shape[0] != 2)
        throw std::runtime_error("Shape of indH2 must be (2, N2)");

    if (indH2buf.shape[1] != valH2buf.size)
        throw std::runtime_error("indH2.shape[1] must match valH2.size");

    if (indH3buf.ndim != 2 || indH3buf.shape[0] != 3)
        throw std::runtime_error("Shape of indH3 must be (3, N3)");

    if (indH3buf.shape[1] != valH3buf.size)
        throw std::runtime_error("indH3.shape[1] must match valH3.size");

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
    
    auto Xout = py::array_t<double, py::array::f_style>({N2, N1});
    py::buffer_info Xoutbuf = Xout.request();
    double *pXout = (double *) Xoutbuf.ptr;

    auto objs = py::array_t<double, py::array::f_style>({1, 1000});
    py::buffer_info objsbuf = objs.request();
    double *pObjs = (double *) objsbuf.ptr;

    double nIter = 0;

    if (adapt <= 0) {
        bcagm3(pIndH1, pValH1, Nt1,
               pIndH2, pValH2, Nt2,
               pIndH3, pValH3, Nt3,
               pX, N1, N2,
               maxIter,
               pXout, pObjs, &nIter);
    } else {
        adapt_bcagm3(pIndH1, pValH1, Nt1,
                     pIndH2, pValH2, Nt2,
                     pIndH3, pValH3, Nt3,
                     pX, N1, N2,
                     maxIter, adapt,
                     pXout, pObjs, &nIter);
    }

    return py::make_tuple(Xout, objs, nIter);
}
