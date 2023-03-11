#ifndef TENSOR_MATCHING_H
#define TENSOR_MATCHING_H

void tensorMatching(double* pX, int N1, int N2,
                          int* pIndH1, double* pValH1, int Nt1 ,
                          int* pIndH2, double* pValH2, int Nt2 ,
                          int* pIndH3, double* pValH3, int Nt3 ,
                          int nIter, int sparse, int stoc,
                          double* pXout, double* pScoreOut);

#endif
