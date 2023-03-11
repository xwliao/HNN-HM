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
//#include "tensorUtil.h"
#include "mathUtil.h"
#include "BCAGM3_QuadSolver.h"
using namespace std;

#define FOR(i, n) for(int i = 0; i < n; ++i)
#define FOR2(i, a, b) for(int i = a; i <= b; ++i)
#define FORD(i, n) for(int i = n-1; i >= 0; i--)
#define FORD2(i, a, b) for(int i = b; i >= a; i--)

/*################################################################## */
// compute gradient of F(xxy) wrt. y
static void computeMultGradient(int* indH3, double* valH3, int Nt3, int* ha, int* hb, double* x, int i, int N1, int N2, double* grad) {
    int dim = N1 * N2;
    FOR(k, dim) grad[k] = 0;
    
    FOR(u, dim) if (x[(1-i)*dim+u] > 0)  {
        FOR(v, dim) if (x[(1-i)*dim+v] > 0) {
            int uv = u*dim + v;
            int a = ha[uv];
            int b = hb[uv];
            if (a < 0) continue;
            FOR2(t, a, b) {
                int w = indH3[t*3+2];
                grad[w] += valH3[t] * x[(1-i)*dim+u] * x[(1-i)*dim+v];
            }
        }
    }
}

/*################################################################## */
static void computeRegulGrad(double* x, int i, int N1, int N2, double eps, int regularizer, double* regulGradi) {
    int dim = N1 * N2;
    double norm = pNorm(x+(1-i)*dim, dim, 1.0);
    double sum = 0;
    FOR(t, dim) regulGradi[t] = 0;
    if (regularizer == 1) {
        FOR(t, dim) {
            double s = 0;
            s += eps*eps*norm*norm;
            s += (1-eps)*(1-eps)*x[(1-i)*dim+t]*x[(1-i)*dim+t];
            s += 2*eps*(1-eps)*norm*x[(1-i)*dim+t];
            regulGradi[t] += (1-eps)*s;
            sum += s;
        }
        FOR(t, dim) regulGradi[t] += eps*sum;
    }
}

/*################################################################## */
// compute gradient of F_alpha(xxy) wrt. y (i-th variable)
// usage i = 1
static void computeBigGrad( int* indH3, double* valH3, int Nt3, int* ha, int* hb,
        int i, double* x, int N1, int N2, double alpha, double eps, int regularizer, double* gradi) {
    int dim = N1 * N2;
    FOR(t, dim) gradi[t] = 0;
    computeMultGradient(indH3, valH3, Nt3, ha, hb, x, i, N1, N2, gradi);
    
    if (alpha > 0) {
        double* regulGradi = new double[dim];
        computeRegulGrad(x, i, N1, N2, eps, regularizer, regulGradi);
        FOR(t, dim) gradi[t] += alpha * regulGradi[t];
        delete[] regulGradi;
    }
}

/*################################################################## */
static void computeRegulMat(double* x, int i, int N1, int N2, double eps, int regularizer, double* regulMati) {
    int dim = N1 * N2;
    double e1 = eps;
    double e2 = e1*e1;
    double e3 = e1*e1*e1;
    double c1 = 1-eps;
    double c2 = c1*c1;
    double c3 = c1*c1*c1;
    FOR(j, dim*dim) regulMati[j] = 0;
    
    double w = 0;
    FOR(j, dim) w += x[(1-i)*dim + j];
    
    if (regularizer == 1) {
        FOR(k, dim) FOR(l, dim) {
            double s = 0;
            s += dim*e3*w + e2*(1-e1)*w*2 + e2*(1-e1)*w;
            s += e1*c2*w*(k==l?1:0) + e1*c2*x[dim*(1-i)+k]
                    + e1*c2*x[dim*(1-i)+l];
            s += c3*(k==l?1:0)*x[dim*(1-i)+k];
            regulMati[dim*k+l] += s;
        }
    }
}
/*################################################################## */
// compute F(..y)
// usage: i = 0
static void computeBigMat(int* indH3, double* valH3, int Nt3,
        int i, double* x, int N1, int N2, double alpha, double eps, int regularizer, double* mat) {
    int dim = N1 * N2;
    FOR(j, dim*dim) mat[j] = 0;
    
    FOR(k, Nt3) {
        int i1 = indH3[k*3];
        int i2 = indH3[k*3+1];
        int i3 = indH3[k*3+2];
        mat[dim*i1+i2] += valH3[k] * x[(1-i)*dim+i3];
    }
    
    if (alpha > 0) {
        double* regulMati = new double[dim*dim];
        computeRegulMat(x, i, N1, N2, eps, regularizer, regulMati);
        FOR(j, dim*dim) mat[j] += alpha * regulMati[j];
        delete[] regulMati;
    }
}
/*################################################################## */
static double getRegulObj(double* x, int N1, int N2, double eps, int regularizer) {
    int dim = N1 * N2;
    double* regulGrad = new double[dim];
    computeRegulGrad(x, 1, N1, N2, eps, regularizer, regulGrad);
    double s = 0;
    FOR(j, dim) s += regulGrad[j]*x[dim+j];
    delete[] regulGrad;
    return s;
}

/*################################################################## */
static double getBigObj(int* indH3, double* valH3, int Nt3, int* ha, int* hb,
        double* x, int N1, int N2, double alpha, double eps, int regularizer) {
    int dim = N1 * N2;
    double* grad = new double[dim];
    computeBigGrad(indH3, valH3, Nt3, ha, hb, 1, x, N1, N2, alpha, eps, regularizer, grad);
    double s = 0;
    FOR(j, dim) s += grad[j]*x[dim+j];
    delete[] grad;
    return s;
}

/*################################################################## */
static void maxPooling(const double* mat, int N1, int N2, double* v) {
    int dim = N1 * N2;
    double* pv = new double[dim];
    double* ppv = new double[dim];
    FOR(i, dim) v[i] = pv[i] = ppv[i] = 1.0 / dim;
    double tol = 1e-12;
    double diff = 1;
    int maxIter = 30;
    int iter = 0;
    double oldf = 0;
    double curf = 0;
    
    while (diff > tol && iter < maxIter)  {
        iter++;
        FOR(i, dim) ppv[i] = pv[i];
        FOR(i, dim) pv[i] = v[i];
        FOR(i, dim) v[i] = 0;
        
        FOR(i, dim) {
            v[i] = pv[i] * mat[i*dim + i];
            FOR(j1, N1) {
                double best = 0;
                FOR(j2, N2) {
                    int j = j1*N2 + j2;
                    double s = mat[i*dim + j] * pv[j];
                    if (best < s) best = s;
                }
                v[i] += best;
            }
        }
        double s = 0;
        FOR(i, dim) s += v[i]*v[i];
        s = sqrt(s);
        FOR(i, dim) v[i] /= s;
        
        double diff1 = 0, diff2 = 0;
        FOR(i, dim) diff1 += (v[i]-pv[i])*(v[i]-pv[i]);
        FOR(i, dim) diff2 += (v[i]-ppv[i])*(v[i]-ppv[i]);
        diff = (diff1 < diff2) ?diff1 :diff2;
    }
    /*printf("MaxPooling finised in %d iterations\n", iter);*/
    delete[] ppv;
    delete[] pv;
}

//##################################################################
static void ipfp(const double* mat, const double* x, int N1, int N2, double* v) {
    int dim = N1 * N2;
    double* y = new double[dim];
    /*FOR(i, dim) y[i] = x[i];*/
    FOR(i, dim) y[i] = 1.0 / dim;
    double* grad = new double[dim];
    double* b = new double[dim];
    double* prev_y = new double[dim];
    double tol = 1e-12;
    int nIter = 0;
    
    double opt = 0;
    FOR(i, dim) v[i] = x[i];
    FOR(k, dim) FOR(l, dim) opt += mat[k*dim+l]*v[k]*v[l];
    
    while (nIter < 20) {
        FOR(i, dim) prev_y[i] = y[i];
        nIter++;
        FOR(j, dim) grad[j] = 0;
        FOR(j, dim) {
            FOR(k, dim) {
                grad[j] += mat[j*dim + k] * y[k];
            }
        }
        discritize(grad, N1, N2, b);
        double C = 0, D = 0, f = 0;
        FOR(k, dim) FOR(l, dim) {
            C += mat[k*dim+l]*y[k]*(b[l]-y[l]);
            D += mat[k*dim+l]*(b[k]-y[k])*(b[l]-y[l]);
            f += mat[k*dim+l]*b[k]*b[l];
        }
        if (D >= 0) {
            FOR(i, dim) y[i] = b[i];
        } else {
            double r = min(-C/D, 1.0);
            FOR(i, dim) y[i] += r * (b[i]-y[i]);
        }
        if (opt < f) {
            opt = f;
            FOR(i, dim) v[i] = b[i];
        }
        double diff = 0;
        FOR(i, dim) diff = max(diff, fabs(y[i]-prev_y[i]));
        if (diff < tol) break;
    }
    /*printf("IPFP finised in %d iterations\n", nIter);*/
    delete[] grad;
    delete[] b;
    delete[] y;
    delete[] prev_y;
}

//##################################################################
static int isHomogeneous(double* x, int dim) {
    double diff = 0;
    FOR(j, dim) {
        diff = max(diff, fabs(x[j] - x[dim+j]));
    }
    return (diff < 1e-12) ?1 :0;
}

/*################################################################## */
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

/*################################################################## */
// subroutine: 1-MP with projection, 2-IPFP
void bcagm3_quad(int* indH3, double* pvalH3, int Nt3,
        double* x0, int N1, int N2, int subroutine,
        double* xout, double* objs, double* nIter) {
    // epsilon parameter used in the convexification of the obj
    double eps = 1.0 / 3;
    
    double* valH3 = new double[Nt3];
    FOR(i, Nt3) valH3[i] = pvalH3[i];
    
    // normalize tensors
    normalize(valH3, Nt3);
    
    double tol = 1e-16;
    int dim = N1*N2;
    int L = dim * 2;
    
    double* x = new double[L];
    FOR(j, L) x[j] = fabs(x0[j]);
    normalize(x, L);
    
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
    
    double* xnew = new double[L];
    double* tmp = new double[L];
    double* mat = new double[dim*dim];
    double* var = new double[dim];
    double* y = new double[L];
    double* grad = new double[dim];
    *nIter = 0;
    double bound = getBound(indH3, valH3, Nt3);
    double alpha = 0;
    bool finish = false;
    
    int nObjs = 0;
    FOR(i, 9) objs[i] = -1;
    double oldf = 0, curf = 0;
    int regularizer = 1;
    
    while (*nIter < 20) {
        oldf = curf;
        *nIter += 1;
        
        // QAP step
        computeBigMat(indH3, valH3, Nt3, 0, x, N1, N2, alpha, eps, regularizer, mat);
        if (subroutine == 1) {
            maxPooling(mat, N1, N2, var);
            // project back to discrete constraint
            FOR(j, dim) tmp[j] = var[j];
            discritize(tmp, N1, N2, var);
        } else {
            ipfp(mat, x, N1, N2, var);
        }
        
        // LAP step
        FOR(k, L) y[k] = var[k%dim];
        computeBigGrad(indH3, valH3, Nt3, ha, hb, 1, y, N1, N2, alpha, eps, regularizer, grad);
        discritize(grad, N1, N2, tmp);
        
        FOR(j, dim) y[j] = var[j];
        FOR(j, dim) y[dim+j] = tmp[j];
        //double curf = getBigObj(indH3, valH3, Nt3, ha, hb, y, N1, N2, alpha, eps);
        curf = 0;
        FOR(j, dim) curf += grad[j] * tmp[j];
        if (curf > oldf) {
            FOR(j, L) x[j] = y[j];
        } else {
            curf = oldf;
        }
        
        if (curf - oldf < tol) {
            if (alpha == 0.0) {
                double best = 0;
                FOR(i, 2) {
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
                FOR(i, 2) {
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
                if (best > curf) {printf("IMPOSSIBLE: bcagm_mp_proj ##################\n");}
                FOR(j, L) x[j] = xnew[j];
                *nIter += 1;
                curf = best;
                printf("bcagm3_quad, subroutine %d: JUMPS ALPHA ON\n", subroutine);
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
    
    delete[] valH3;
    delete[] x;
    delete[] ha;
    delete[] hb;
    delete[] tmp;
    delete[] xnew;
    delete[] mat;
    delete[] var;
    delete[] y;
    delete[] grad;
}

/*################################################################## */
// subroutine: 1-MP with projection, 2-IPFP
void adapt_bcagm3_quad(int* indH3, double* pvalH3, int Nt3,
        double* x0, int N1, int N2, int subroutine, int regularizer,
        double* xout, double* objs, double* nIter) {
    // epsilon parameter used in the convexification of the obj
    double eps = 1.0 / 3;
    
    double* valH3 = new double[Nt3];
    FOR(i, Nt3) valH3[i] = pvalH3[i];
    
    // normalize tensors
    normalize(valH3, Nt3);
    
    double tol = 1e-16;
    int dim = N1*N2;
    int L = dim * 2;
    
    double* x = new double[L];
    FOR(j, L) x[j] = fabs(x0[j]);
    normalize(x, L);
    
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
    
    double* xnew = new double[L];
    double* tmp = new double[L];
    double* mat = new double[dim*dim];
    double* var = new double[dim];
    double* y = new double[L];
    double* grad = new double[dim];
    *nIter = 0;
    double bound = getBound(indH3, valH3, Nt3);
    double alpha = 0;
    bool finish = false;
    
    int nObjs = 0;
    FOR(i, 9) objs[i] = -1;
    double oldf = 0, curf = 0;
    
    while (*nIter < 20) {
        oldf = curf;
        *nIter += 1;
        
        // QAP step
        computeBigMat(indH3, valH3, Nt3, 0, x, N1, N2, alpha, eps, regularizer, mat);
        if (subroutine == 1) {
            maxPooling(mat, N1, N2, var);
            // project back to discrete constraint
            FOR(j, dim) tmp[j] = var[j];
            discritize(tmp, N1, N2, var);
        } else {
            ipfp(mat, x, N1, N2, var);
        }
        
        // LAP step
        FOR(k, L) y[k] = var[k%dim];
        computeBigGrad(indH3, valH3, Nt3, ha, hb, 1, y, N1, N2, alpha, eps, regularizer, grad);
        discritize(grad, N1, N2, tmp);
        
        FOR(j, dim) y[j] = var[j];
        FOR(j, dim) y[dim+j] = tmp[j];
        //double curf = getBigObj(indH3, valH3, Nt3, ha, hb, y, N1, N2, alpha, eps);
        curf = 0;
        FOR(j, dim) curf += grad[j] * tmp[j];
        if (curf > oldf) {
            FOR(j, L) x[j] = y[j];
        } else {
            curf = oldf;
        }
        
        if (curf - oldf < tol) {
            double best = 0;
            FOR(i, 2) {
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
            } else {
                if (nObjs <= 6) {
                    objs[nObjs++] = isHomogeneous(x, dim);
                    objs[nObjs++] = curf;
                    objs[nObjs++] = best;
                }
                
                double xxy = getRegulObj(x, N1, N2, eps, regularizer);
                FOR(j, L) tmp[j] = x[j%dim];
                double uuu = getRegulObj(tmp, N1, N2, eps, regularizer);
                double inc = (curf-best) / (uuu-xxy) + 1e-8;
                alpha = alpha + inc;
                
                curf += inc * xxy;
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
    
    delete[] valH3;
    delete[] x;
    delete[] ha;
    delete[] hb;
    delete[] tmp;
    delete[] xnew;
    delete[] mat;
    delete[] var;
    delete[] y;
    delete[] grad;
}
