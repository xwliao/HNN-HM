#include <iostream>
#include <exception>
#include <time.h>
#include <stdlib.h>
#include <float.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <algorithm>
#include "mathUtil.h"
//#include "tensorUtil.h"
#include "BCAGM3_LinSolver.h"
using namespace std;

#define FOR(i, n) for(int i = 0; i < n; ++i)
#define FOR2(i, a, b) for(int i = a; i <= b; ++i)
#define FORD(i, n) for(int i = n-1; i >= 0; i--)
#define FORD2(i, a, b) for(int i = b; i >= a; i--)

// compute gradient of the original obj w.r.t. the last variable stored in x[dim*3]
static void computeMultGradient(int* indH3, double* valH3, int Nt3, int* ha, int* hb,
        double* x, int N1, int N2, double* grad) {
    int dim = N1 * N2;
    FOR(k, dim) grad[k] = 0;
    FOR(i, dim) if (x[i] > 0)  {
        FOR(j, dim) if (x[dim+j] > 0) {
            int ij = i*dim + j;
            int a = ha[ij];
            int b = hb[ij];
            if (a < 0) continue;
            FOR2(t, a, b) {
                int k = indH3[t*3+2];
                grad[k] += valH3[t] * x[i] * x[dim+j];
            }
        }
    }
}

// regularizer: 1)standard, 2)sigma_i xiyizi, 3)<1,x><y,z> + <1,y><x,z> + <1,z><x,y>
// eps is a parameter of standard regularizer
static void computeRegulGrad(double* x, int i, int N1, int N2, double eps, int regularizer, double* regulGradi) {
    int dim = N1 * N2;
    FOR(t, dim) regulGradi[t] = 0;
    double norms[3];
    FOR(j, 3) norms[j] = pNorm(x+j*dim, dim, 1.0);
    int j = (i+1) % 3;
    int k = (i+2) % 3;
    if (regularizer == 1) {
        double sum = 0;
        FOR(t, dim) {
            double s = 0;
            s += eps*eps*norms[j]*norms[k];
            s += (1-eps)*(1-eps)*x[j*dim+t]*x[k*dim+t];
            s += eps*(1-eps)*norms[j]*x[k*dim+t];
            s += eps*(1-eps)*norms[k]*x[j*dim+t];
            regulGradi[t] += (1-eps)*s;
            sum += s;
        }
        FOR(t, dim) regulGradi[t] += eps*sum;
    } else if (regularizer == 2) {
        double inner = 0;
        FOR(t, dim) inner += x[dim*j+t] * x[dim*k+t];
        FOR(t, dim) regulGradi[t] += inner + norms[j]*x[dim*k+t] + norms[k]*x[dim*j+t];
    } else if (regularizer == 3) {
        FOR(t, dim) regulGradi[t] += x[dim*j+t]*x[dim*k+t];
    }
}

// compute gradient of the modified obj w.r.t. the i-th variable stored in x[dim*3]
static void computeBigGrad( int* indH3, double* valH3, int Nt3, int* ha, int* hb,
        int i, double* x, int N1, int N2, double alpha, double eps, int regularizer, double* gradi) {
    int dim = N1 * N2;
    FOR(t, dim) gradi[t] = 0;
    double* y = new double[dim*3];
    int n = 0;
    FOR2(k, i+1, i+3) {
        FOR(t, dim) y[n++] = x[(k%3)*dim + t];
    }
    computeMultGradient(indH3, valH3, Nt3, ha, hb, y, N1, N2, gradi);
    delete[] y;
    
    if (alpha > 0) {
        double* regulGradi = new double[dim];
        computeRegulGrad(x, i, N1, N2, eps, regularizer, regulGradi);
        FOR(t, dim) gradi[t] += alpha * regulGradi[t];
        delete[] regulGradi;
    }
}

static double getRegulObj(double* x, int N1, int N2, double eps, int regularizer) {
    int dim = N1*N2;
    double* regulGradi = new double[dim];
    int i = 0;
    computeRegulGrad(x, i, N1, N2, eps, regularizer, regulGradi);
    double s = 0;
    FOR(t, dim) s += x[dim*i+t] * regulGradi[t];
    delete[] regulGradi;
    return s;
}

static double getBigObj( int* indH3, double* valH3, int Nt3, int* ha, int* hb,
        double* x, int N1, int N2, double alpha, double eps, int regularizer) {
    int dim = N1 * N2;
    double* gradi = new double[dim];
    int i = 0;
    computeBigGrad(indH3, valH3, Nt3, ha, hb, i, x, N1, N2, alpha, eps, regularizer, gradi);
    double s = 0;
    FOR(j, dim) s += gradi[j] * x[i*dim + j];
    delete[] gradi;
    return s;
}

static int isHomogeneous(double* x, int dim) {
    double diff = 0;
    FOR(i, 2) {
        FOR(j, dim) {
            diff = max(diff, fabs(x[i*dim+j] - x[2*dim+j]));
        }
    }
    return (diff < 1e-12) ?1 :0;
}

static double getBound(int* indH3, double* valH3, int Nt3) {
    double bound = 0;
    //FOR(i, Nt3) bound += valH3[i] * valH3[i];
    //bound = (27.0/4)*sqrt(bound);
    double s = valH3[0]*valH3[0];
    bound = s;
    FOR2(t, 1, Nt3-1) {
        if (indH3[3*t]!=indH3[3*(t-1)]) {
            bound = (bound < s) ?s :bound;
            s = valH3[t]*valH3[t];
        } else {
            s += valH3[t]*valH3[t];
        }
    }
    bound = (bound < s) ?s :bound;
    return sqrt(bound);
}

void bcagm3(	int* indH1, double* pvalH1, int Nt1,
        int* indH2, double* pvalH2, int Nt2,
        int* indH3, double* pvalH3, int Nt3,
        double* x0, int N1, int N2, int maxIter,
        double* xout, double* objs, double* nIter) {
    // epsilon parameter used in the convexification of the obj
    double eps = 1.0 / 3;
    
    double* valH3 = new double[Nt3];
    FOR(i, Nt3) valH3[i] = pvalH3[i];
    
    // normalize tensors
    normalize(valH3, Nt3);
    
    double tol = 1e-16;
    int dim = N1*N2;
    int L = dim * 3;
    
    double* x = new double[L];
    FOR(j, L) x[j] = fabs(x0[j]);
    normalize(x, L);
    
    double* xnew = new double[L];
    double* tmp = new double[L];
    double* var = new double[dim];
    double* gradi = new double[dim];
    
    // create mapping for nonzeros
    int L2 = dim*dim;
    int* ha = new int[L2];
    int* hb = new int[L2];
    FOR(i, L2) ha[i] = hb[i] = -1;
    FOR(t, Nt3) {
        int i = indH3[3*t];
        int j = indH3[3*t+1];
        int k = i*dim + j;
        if (ha[k] == -1) ha[k] = t;
        hb[k] = t;
    }
    
    double bound = getBound(indH3, valH3, Nt3);
    double alpha = 0;
    double start;
    bool finish = false;
    *nIter = 0;
    int nObjs = 0;
    FOR(i, 9) objs[i] = -1;
    double oldf = 0, curf = 0;
    int regularizer = 1;
    
    while (*nIter < maxIter) {
        oldf = curf;
        *nIter += 1;
        
        FOR(i, 3) {
            computeBigGrad(indH3, valH3, Nt3, ha, hb, i, x, N1, N2, alpha, eps, regularizer, gradi);
            discritize(gradi, N1, N2, var);
            FOR(j, dim) x[i*dim+j] = var[j];
        }
        
        curf = 0;
        FOR(j, dim) curf += gradi[j] * var[j];
        //double curf = getBigObj(indH3, valH3, Nt3, ha, hb, x, N1, N2, alpha, eps);
        if (curf < oldf) printf("ERROR: BCAGM3 GOT DESCENT !!!!!!!!!!!\n");
        
        if (curf - oldf < tol) {
            if (alpha == 0.0) {
                double best = 0;
                FOR(i, 3) {
                    FOR(j, L) tmp[j] = x[i*dim + j%dim];
                    double f = getBigObj(indH3, valH3, Nt3, ha, hb, tmp, N1, N2, alpha, eps, regularizer);
                    if (best < f) {
                        best = f;
                        FOR(j, L) xnew[j] = tmp[j];
                    }
                }
                if (nObjs == 0) {
                    objs[nObjs++] = isHomogeneous(x, dim);
                    objs[nObjs++] = curf;
                    objs[nObjs++] = best;
                }
                if (curf < best) {
                    FOR(j, L) x[j] = xnew[j];
                    *nIter += 1;
                    curf = best;
                } else {
                    objs[nObjs++] = isHomogeneous(x, dim);
                    objs[nObjs++] = curf;
                    objs[nObjs++] = best;
                    
                    alpha = bound;
                    curf = getBigObj(indH3, valH3, Nt3, ha, hb, x, N1, N2, alpha, eps, regularizer);
                    objs[nObjs++] = curf;
                }
                if (isHomogeneous(x, dim)) {
                    FOR(j, dim) xout[j] = x[j];
                    finish = true;
                    break;
                }
            } else {
                double best = 0;
                FOR(i, 3) {
                    FOR(j, L) tmp[j] = x[i*dim + j%dim];
                    double f = getBigObj(indH3, valH3, Nt3, ha, hb, tmp, N1, N2, alpha, eps, regularizer);
                    if (best < f) {
                        best = f;
                        FOR(j, L) xnew[j] = tmp[j];
                    }
                }
                if (best == curf) {
                    FOR(j, dim) xout[j] = xnew[j];
                    finish = true;
                    break;
                }
                if (best > curf) {printf("IMPOSSIBLE: bcagm3 #####################################\n");}
                FOR(j, L) x[j] = xnew[j];
                *nIter += 1;
                curf = best;
                printf("bcagm3: JUMPS ALPHA ON\n");
            }
        }
        if (finish) break;
    }
    
    if (!finish) {
        FOR(j, dim) xout[j] = x[j];
    }
    
    objs[nObjs++] = curf;
    FOR(j, L) tmp[j] = xout[j%dim];
    objs[nObjs++] = getBigObj(indH3, valH3, Nt3, ha, hb, tmp, N1, N2, 0, eps, regularizer);
    
    delete[] gradi;
    delete[] var;
    delete[] tmp;
    delete[] xnew;
    delete[] x;
    delete[] valH3;
    delete[] ha;
    delete[] hb;
}


// Adaptive BCAGM3 Methods
void adapt_bcagm3(int* indH1, double* pvalH1, int Nt1,
        int* indH2, double* pvalH2, int Nt2,
        int* indH3, double* pvalH3, int Nt3,
        double* x0, int N1, int N2, int maxIter, int regularizer,
        double* xout, double* objs, double* nIter) {
    // epsilon parameter used in the convexification of the obj
    double eps = 1.0 / 3;
    
    double* valH3 = new double[Nt3];
    FOR(i, Nt3) valH3[i] = pvalH3[i];
    
    // normalize tensors
    normalize(valH3, Nt3);
    
    double tol = 1e-16;
    int dim = N1*N2;
    int L = dim * 3;
    
    double* x = new double[L];
    FOR(j, L) x[j] = fabs(x0[j]);
    normalize(x, L);
    
    double* xnew = new double[L];
    double* tmp = new double[L];
    double* var = new double[dim];
    double* gradi = new double[dim];
    
    // create mapping for nonzeros
    int L2 = dim*dim;
    int* ha = new int[L2];
    int* hb = new int[L2];
    FOR(i, L2) ha[i] = hb[i] = -1;
    FOR(t, Nt3) {
        int i = indH3[3*t];
        int j = indH3[3*t+1];
        int k = i*dim + j;
        if (ha[k] == -1) ha[k] = t;
        hb[k] = t;
    }
    
    double bound = getBound(indH3, valH3, Nt3);
    double alpha = 0;
    bool finish = false;
    *nIter = 0;
    int nObjs = 0;
    FOR(i, 9) objs[i] = -1;
    double oldf = 0, curf = 0;
    
    while (*nIter < maxIter) {
        oldf = curf;
        *nIter += 1;
        
        FOR(i, 3) {
            computeBigGrad(indH3, valH3, Nt3, ha, hb, i, x, N1, N2, alpha, eps, regularizer, gradi);
            discritize(gradi, N1, N2, var);
            FOR(j, dim) x[i*dim+j] = var[j];
        }
        
        curf = 0;
        FOR(j, dim) curf += gradi[j] * var[j];
        //double curf = getBigObj(indH3, valH3, Nt3, ha, hb, x, N1, N2, alpha, eps);
        if (curf < oldf) printf("ERROR: adapt_bcagm3 GOT DESCENT !!!!!!!!!!!\n");
        
        if (curf - oldf < tol) {
            // compute the best homogeneous iterate
            double best = 0;
            FOR(i, 3) {
                FOR(j, L) tmp[j] = x[i*dim + j%dim];
                double f = getBigObj(indH3, valH3, Nt3, ha, hb, tmp, N1, N2, alpha, eps, regularizer);
                if (best < f) {
                    best = f;
                    FOR(j, L) xnew[j] = tmp[j];
                }
            }
            if (nObjs == 0) {
                // store result of pure BCA
                objs[nObjs++] = isHomogeneous(x, dim);
                objs[nObjs++] = curf;
                objs[nObjs++] = best;
            }
            // check critical point
            if (isHomogeneous(x, dim)) {
                FOR(j, dim) xout[j] = x[j];
                finish = true;
                break;
            }
            if (curf < best) {
                FOR(j, L) x[j] = xnew[j];
                *nIter += 1;
                curf = best;
                printf("adapt_bcagm3: JUMPS ALPHA ON\n");
            } else {
                if (nObjs <= 6) {
                    objs[nObjs++] = isHomogeneous(x, dim);
                    objs[nObjs++] = curf;
                    objs[nObjs++] = best;
                }
                
                double xyz = getRegulObj(x, N1, N2, eps, regularizer);
                FOR(j, L) tmp[j] = x[j%dim];
                double uuu = getRegulObj(tmp, N1, N2, eps, regularizer);
                double inc = (curf-best) / (uuu-xyz) + 1e-8;
                alpha = alpha + inc;
                
                //FOR(j, L) x[j] = xnew[j];
                
                //curf = getBigObj(indH3, valH3, Nt3, ha, hb, x, N1, N2, alpha, eps, regularizer);
                curf += inc * xyz;
                if (nObjs <= 6) objs[nObjs++] = curf;
            }
        }
        if (finish) break;
    }
    
    if (!finish) {
        FOR(j, dim) xout[j] = x[j];
    }
    
    objs[7] = curf;
    FOR(j, L) tmp[j] = xout[j%dim];
    objs[8] = getBigObj(indH3, valH3, Nt3, ha, hb, tmp, N1, N2, 0, eps, regularizer);
    
    delete[] gradi;
    delete[] var;
    delete[] tmp;
    delete[] xnew;
    delete[] x;
    delete[] valH3;
    delete[] ha;
    delete[] hb;
}
