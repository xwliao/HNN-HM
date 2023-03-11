#ifndef COMPUTE_FEATURE_H
#define COMPUTE_FEATURE_H

/* feat1 feat2 <- P1 P2 t1 type */

void computeFeatureSimple(double* pP, int i , int j, int k , double* pF);
void computeFeatureSimple2(double* pP , int nP, int* pT , int nT , double* pF);
void computeFeature(double* pP1 , int nP1 , double* pP2 , int nP2 ,
                    int* pT1 , int nT1 , double* pF1 , double* pF2);

#endif
