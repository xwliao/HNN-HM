/**
 * Demo Code for BCAGM+Psi algorithm (CVPR 2015)
 *
 * Quynh Nguyen, Antoine Gautier, Matthias Hein
 * A Flexible Tensor Block Coordinate Ascent Scheme for Hypergraph Matching
 * Proc. of the IEEE International Conference on Computer Vision and Pattern Recognition (CVPR 2015)
 *
 * Please cite our work if you find this code useful for your research
 *
 * written by Quynh Nguyen, 2014, Saarland University, Germany
 * http://www.ml.uni-saarland.de/people/nguyen.htm
 *
 * Version: 1.0
 * Date: 28.04.2015
 **/
#ifndef BCAGM_QUAD_SOLVER_H
#define BCAGM_QUAD_SOLVER_H 1

/*################################################################## */
// BCAGM+MP algorithm
void bcagm_quad(	int* indH3, double* pvalH3, int Nt3,
        double* x0, int N1, int N2, int subroutine,
        double* xout, double* objs, double* nIter);

/*################################################################## */
// BCAGM+MP algorithm
void adapt_bcagm_quad(	int* indH3, double* pvalH3, int Nt3,
        double* x0, int N1, int N2, int subroutine, int regularizer,
        double* xout, double* objs, double* nIter);

#endif
