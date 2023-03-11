#ifndef RRWHM_H
#define RRWHM_H

void RRWM(double* pX, int N1, int N2,
          int* pIndH1, double* pValH1, int Nt1,
          int* pIndH2, double* pValH2, int Nt2,
          int* pIndH3, double* pValH3, int Nt3,
          int nIter, double* pC,
          double* pXout);

#endif
