#include <stdlib.h>
#include "util.h"

double** array_to_matrix(const double* m, int rows, int cols) {
  int i,j;
  double** r;
  r = (double**)calloc(rows,sizeof(double*));
  for(i=0;i<rows;i++)
  {
    r[i] = (double*)calloc(cols,sizeof(double));
    for(j=0;j<cols;j++)
      r[i][j] = m[i*cols+j];
  }
  return r;
}
