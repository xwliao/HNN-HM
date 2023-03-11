#ifndef ADGM_H
#define ADGM_H

/********************************************************************
 ********************************************************************
 Alternating Direction Graph Matching
 Version: 0.1 (pre-release)
 Written by: D. Khuê Lê-Huu (khue.le@centralesupelec.fr)
 
 If you use any part of this code, please cite:
 
D. Khuê Lê-Huu and Nikos Paragios. Alternating Direction Graph Matching. IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2017.

BibTeX:

@inproceedings{lehuu2017adgm,
 title={Alternating Direction Graph Matching},
 author={L{\^e}-Huu, D. Khu{\^e} and Paragios, Nikos},
 booktitle = {Proceedings of the {IEEE} Conference on Computer Vision and Pattern Recognition ({CVPR})},
 year = {2017}
}


*********************
NOTE: 

- This is a preliminary re-implementation in C++ Eigen, 
I haven't tested it on the full benchmark yet and thus the performance 
is not guaranteed (in particular some caching part has not been implemented
yet, so it should be slower). Before using this software, please check if 
any updated version is available on my website www.khue.fr.
- In the current version, only third-order potentials are supported.

 ** This file is distributed in the hope that it will be useful,
 ** but WITHOUT ANY WARRANTY; without even the implied 
 ** warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  
 **
 ********************************************************************
 ********************************************************************/
 
#define NONE 0
#define EQUALITY 1
#define INEQUALITY 2
#define ROW 1
#define COL 2
#define VARIANT_1 1
#define VARIANT_2 2

int ADGM_3rdORDER(double* x, double* residuals, double &rho, const double* _x0, int n1, int n2,
                  const int* indH3, const double* _valH3, int Nt3, const double* rhos, int nrho,
                  int MAX_ITER, bool verb, bool restart, int iter1, int iter2, int variant);

int ADGM_3rdORDER_SYMMETRIC(double* x, double* residuals, double &rho, const double* _x0, int n1, int n2,
                            const int* _indH3, const double* _valH3, int _Nt3, const double* rhos, int nrho,
                            int MAX_ITER, bool verb, bool restart, int iter1, int iter2, int variant);

#endif // ADGM_H
