#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "RRWHM.h"

double getMax(double* pData, int nData);
void inflate(double* pDataIn, double* pDataOut, int nData, double beta);
void normalize(double* pData, int nData);

void RRWM(double* pX, int N1, int N2,
                          int* pIndH1, double* pValH1, int Nt1 ,
                          int* pIndH2, double* pValH2, int Nt2 ,
                          int* pIndH3, double* pValH3, int Nt3 ,
                          int nIter, double* pC,
                          double* pXout)
{   
    int n1, n2;
	int n,t;
    
    // Parameters
    double deltaMin = pow(0.1, 12);
    double beta = 30;

    int NN=N1*N2;
    // Memory Allocation
	double* pXnew = new double[NN];
    double* pY = new double[NN];
    double* pTemp = new double[NN];
    double* pValH = new double[Nt3];

    // Get Maximum sum value (element-wise)
    for(n = 0; n < NN; n++)
        pXnew[n] = 0;
    for(t = 0; t < Nt3; t++)
    {
        pXnew[pIndH3[t]] += pValH3[t];
        pXnew[pIndH3[t+Nt3]] += pValH3[t];
        pXnew[pIndH3[t+2*Nt3]] += pValH3[t];
    }
    double Hmax = getMax(pXnew, NN);    
    
    // Normalize the Tensor
    for(t = 0; t < Nt3; t++)
        pValH[t] = pValH3[t]/Hmax;
    
    // Normalize the initial vector
    normalize(pX, NN);
    
    // Eliminate Conflictions???
    
    // Main Loop
    int nIterCount = 0;
    double delta = deltaMin+1;
    double sum;
    
    while(nIterCount < nIter && delta > pow(0.1, 6))
    {
        nIterCount++;
        for(n = 0; n < NN; n++)
            pXnew[n] = 0;
        for(t = 0; t < Nt3; t++)
        {
            pXnew[pIndH3[t]] += pValH[t] * pX[pIndH3[t+Nt3]] * pX[pIndH3[t+2*Nt3]];
            pXnew[pIndH3[t+Nt3]] += pValH[t] * pX[pIndH3[t]] * pX[pIndH3[t+2*Nt3]];
            pXnew[pIndH3[t+2*Nt3]] += pValH[t] * pX[pIndH3[t]] * pX[pIndH3[t+Nt3]];
        }
        normalize(pXnew, NN);
        
        // Inflate
        inflate(pXnew, pY, NN, beta);
                
        // Bistochastic Normalization
        // Did I cover N1 ~= N2 case?
        delta = deltaMin+1;
        while(delta > deltaMin)
        {
            for(n = 0; n < NN; n++)
                pTemp[n] = pY[n];
            
            for(n1 = 0; n1 < N1; n1++)
            {
                sum = 0;
                for(n2 = 0; n2 < N2; n2++)
                    sum += pY[n1+n2*N1];
                for(n2 = 0; n2 < N2; n2++)
                    pY[n1+n2*N1] /= sum;
            }
            for(n2 = 0; n2 < N2; n2++)
            {
                sum = 0;
                for(n1 = 0; n1 < N1; n1++)
                    sum += pY[n1+n2*N1];
                for(n1 = 0; n1 < N1; n1++)
                    pY[n1+n2*N1] /= sum;
            }
            
            delta = 0;
            for(n = 0; n < NN; n++)
            	delta += (pY[n]-pTemp[n])*(pY[n]-pTemp[n]);
            
            for(n = 0; n < NN; n++)
                pTemp[n] = pY[n];
            
        }
        normalize(pY, NN);        
        
        for(n = 0; n < NN; n++)
            pXout[n] = pXnew[n] * (*pC) + (1-(*pC)) * pY[n];
        
        normalize(pXout, NN);
               
        delta = 0;
		for(n = 0; n < NN; n++)
			delta += (pXout[n]-pX[n])*(pXout[n]-pX[n]);
        
		for(n = 0; n < NN; n++)
			pX[n] = pXout[n];
                
   }
    
    delete[] pXnew;
    delete[] pY;
    delete[] pTemp;
    delete[] pValH;
    
}

void normalize(double* pData, int nData)
{
    double sum = 0;
    int i;
    for(i = 0; i < nData; i++)
        sum += pData[i];
    for(i = 0; i < nData; i++)
        pData[i] /= sum;
}

double getMax(double* pData, int nData)
{
    double maxValue = pData[0];
    for(int i = 1; i < nData; i++)
        if(pData[i] > maxValue)
            maxValue = pData[i];
    return maxValue;
}

void inflate(double* pDataIn, double* pDataOut, int nData, double beta)
{
    // Find maximum value
    double maxData = getMax(pDataIn, nData);
    double amp = beta/maxData;
    
    for(int i = 0; i < nData; i++)
        pDataOut[i] = exp(amp*pDataIn[i]);
}
