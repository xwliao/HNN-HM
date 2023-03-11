#include <iostream>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "ADGM.h"

namespace py = pybind11;

py::tuple wrapADGM3rdOrder(
        py::array_t<double, py::array::f_style | py::array::forcecast> X,
        py::array_t<int, py::array::f_style | py::array::forcecast> indH1, py::array_t<double, py::array::f_style | py::array::forcecast> valH1,
        py::array_t<int, py::array::f_style | py::array::forcecast> indH2, py::array_t<double, py::array::f_style | py::array::forcecast> valH2,
        py::array_t<int, py::array::f_style | py::array::forcecast> indH3, py::array_t<double, py::array::f_style | py::array::forcecast> valH3,
        py::array_t<double, py::array::f_style | py::array::forcecast> rhos,
        int max_iter, bool verb, bool restart, int iter1, int iter2, int variant)
{
    if (verb) {
        std::cout << "Started C++ ADGM..." << std::endl;	
    }

    py::buffer_info Xbuf = X.request();

    py::buffer_info indH1buf = indH1.request();
    py::buffer_info valH1buf = valH1.request();

    py::buffer_info indH2buf = indH2.request();
    py::buffer_info valH2buf = valH2.request();

    py::buffer_info indH3buf = indH3.request();
    py::buffer_info valH3buf = valH3.request();

    py::buffer_info rhosbuf = rhos.request();

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

    int Nrho = rhosbuf.size;

    double *pX = (double *) Xbuf.ptr;

    int *pIndH1 = (int *) indH1buf.ptr;
    double *pValH1 = (double *) valH1buf.ptr;

    int *pIndH2 = (int *) indH2buf.ptr;
    double *pValH2 = (double *) valH2buf.ptr;

    int *pIndH3 = (int *) indH3buf.ptr;
    double *pValH3 = (double *) valH3buf.ptr;

    double *pRhos = (double *) rhosbuf.ptr;
    
    auto Xout = py::array_t<double, py::array::f_style>(Xbuf.shape);
    py::buffer_info Xoutbuf = Xout.request();
    double *pXout = (double *) Xoutbuf.ptr;

    double* pResidualsTemp = new double[max_iter];

    double rho = 0; // the final rho will be returned as well

    int N = ADGM_3rdORDER(pXout, pResidualsTemp, rho,
                          pX, N2, N1,
                          pIndH3, pValH3, Nt3,
                          pRhos, Nrho,
                          max_iter, verb, restart, iter1, iter2, variant);

    auto residuals = py::array_t<double, py::array::f_style>({N, 1});
    if (pResidualsTemp != NULL) {
        py::buffer_info residualsbuf = residuals.request();
        double *pResiduals = (double *) residualsbuf.ptr;
        std::copy(pResidualsTemp, pResidualsTemp + N, pResiduals);
        delete[] pResidualsTemp;
        pResidualsTemp = NULL;
    }

    return py::make_tuple(Xout, residuals, rho);
}


py::tuple wrapADGM3rdOrderSymmetric(
        py::array_t<double, py::array::f_style | py::array::forcecast> X,
        py::array_t<int, py::array::f_style | py::array::forcecast> indH1, py::array_t<double, py::array::f_style | py::array::forcecast> valH1,
        py::array_t<int, py::array::f_style | py::array::forcecast> indH2, py::array_t<double, py::array::f_style | py::array::forcecast> valH2,
        py::array_t<int, py::array::f_style | py::array::forcecast> indH3, py::array_t<double, py::array::f_style | py::array::forcecast> valH3,
        py::array_t<double, py::array::f_style | py::array::forcecast> rhos,
        int max_iter, bool verb, bool restart, int iter1, int iter2, int variant)
{
    if (verb) {
        std::cout << "Started C++ ADGM..." << std::endl;	
    }

    py::buffer_info Xbuf = X.request();

    py::buffer_info indH1buf = indH1.request();
    py::buffer_info valH1buf = valH1.request();

    py::buffer_info indH2buf = indH2.request();
    py::buffer_info valH2buf = valH2.request();

    py::buffer_info indH3buf = indH3.request();
    py::buffer_info valH3buf = valH3.request();

    py::buffer_info rhosbuf = rhos.request();

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

    int Nrho = rhosbuf.size;

    double *pX = (double *) Xbuf.ptr;

    int *pIndH1 = (int *) indH1buf.ptr;
    double *pValH1 = (double *) valH1buf.ptr;

    int *pIndH2 = (int *) indH2buf.ptr;
    double *pValH2 = (double *) valH2buf.ptr;

    int *pIndH3 = (int *) indH3buf.ptr;
    double *pValH3 = (double *) valH3buf.ptr;

    double *pRhos = (double *) rhosbuf.ptr;
    
    auto Xout = py::array_t<double, py::array::f_style>(Xbuf.shape);
    py::buffer_info Xoutbuf = Xout.request();
    double *pXout = (double *) Xoutbuf.ptr;

    double* pResidualsTemp = new double[max_iter];

    double rho = 0; // the final rho will be returned as well

    int N = ADGM_3rdORDER_SYMMETRIC(pXout, pResidualsTemp, rho,
                                    pX, N2, N1,
                                    pIndH3, pValH3, Nt3,
                                    pRhos, Nrho,
                                    max_iter, verb, restart, iter1, iter2, variant);

    auto residuals = py::array_t<double, py::array::f_style>({N, 1});
    if (pResidualsTemp != NULL) {
        py::buffer_info residualsbuf = residuals.request();
        double *pResiduals = (double *) residualsbuf.ptr;
        std::copy(pResidualsTemp, pResidualsTemp + N, pResiduals);
        delete[] pResidualsTemp;
        pResidualsTemp = NULL;
    }

    return py::make_tuple(Xout, residuals, rho);
}


PYBIND11_MODULE(ADGMCore, m) {
    m.doc() = "Alternating Direction Graph Matching (ADGM)"; // module docstring
    m.attr("VARIANT_1") = py::int_(VARIANT_1);
    m.attr("VARIANT_2") = py::int_(VARIANT_2);
    m.def("wrapADGM3rdOrder", &wrapADGM3rdOrder, "");
    m.def("wrapADGM3rdOrderSymmetric", &wrapADGM3rdOrderSymmetric, "");
}
