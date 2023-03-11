#ifndef BCAGM_QUAD_SOLVER_H
#define BCAGM_QUAD_SOLVER_H

/*################################################################## */
// subroutine: 1-MP with projection, 2-IPFP
void bcagm3_quad(int* indH3, double* pvalH3, int Nt3,
        double* x0, int N1, int N2, int subroutine,
        double* xout, double* objs, double* nIter);

/*################################################################## */
// subroutine: 1-MP with projection, 2-IPFP
void adapt_bcagm3_quad(int* indH3, double* pvalH3, int Nt3,
        double* x0, int N1, int N2, int subroutine, int regularizer,
        double* xout, double* objs, double* nIter);

#endif
