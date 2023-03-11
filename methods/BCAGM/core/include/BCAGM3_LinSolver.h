#ifndef BCAGM_LIN_SOLVER_H
#define BCAGM_LIN_SOLVER_H

void bcagm3(int* indH1, double* pvalH1, int Nt1,
        int* indH2, double* pvalH2, int Nt2,
        int* indH3, double* pvalH3, int Nt3,
        double* x0, int N1, int N2, int maxIter,
        double* xout, double* objs, double* nIter);


// Adaptive BCAGM3 Methods
void adapt_bcagm3(int* indH1, double* pvalH1, int Nt1,
        int* indH2, double* pvalH2, int Nt2,
        int* indH3, double* pvalH3, int Nt3,
        double* x0, int N1, int N2, int maxIter, int regularizer,
        double* xout, double* objs, double* nIter);

#endif
