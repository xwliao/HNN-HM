/**
 * Demo Code for BCAGM algorithm (CVPR 2015)
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

#ifndef BCAGM_LIN_SOLVER_H
#define BCAGM_LIN_SOLVER_H 1

// main algorithm
// NOTE that the input arrays: problem.indH3 and problem.valH3 should be symmetric themselves.
// This means they should store all the six permutations of each non-zero entries instead of storing only one
void bcagm(	int* indH1, double* pvalH1, int Nt1,
        int* indH2, double* pvalH2, int Nt2,
        int* indH3, double* pvalH3, int Nt3,
        double* x0, int N1, int N2, int maxIter,
        double* xout, double* objs, double* nIter);

// Adaptive BCAGM Methods
void adapt_bcagm(int* indH1, double* pvalH1, int Nt1,
        int* indH2, double* pvalH2, int Nt2,
        int* indH3, double* pvalH3, int Nt3,
        double* x0, int N1, int N2, int maxIter,
        double* xout, double* objs, double* nIter);

#endif
