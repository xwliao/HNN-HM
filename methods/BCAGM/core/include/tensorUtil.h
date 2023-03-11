#ifndef TENSOR_UTIL_H
#define TENSOR_UTIL_H 1

#include <iostream>
#include <ctime>
#include <cstdlib>
#include <cstdio>
#include <sstream>
#include <fstream>
#include <string>
#include <vector>
#include <map>
#include <cmath>
#include <algorithm>
#include "util.h"
#include "hungarian_Q.h"
using namespace std;

#define FOR(i, n) for(int i = 0; i < n; ++i)
#define FOR2(i, a, b) for(int i = a; i <= b; ++i)
#define FORD(i, n) for(int i = n-1; i >= 0; i--)
#define FORD2(i, a, b) for(int i = b; i >= a; i--)

void bistoc(double* x, int n) {
	int dim = n*n;
	double diff;
	double* y = new double[dim];
	double eps = 1e-10;
	int nIter = 0;

	while (1) {
		FOR(k, dim) y[k] = x[k];
		FOR(i, n) {
			double s = 0;
			FOR(j, n) s += fabs(x[i*n + j]);
			if (s != 0) {
				FOR(j, n) x[i*n + j] /= s;
			}
		}

		FOR(j, n) {
			double s = 0;
			FOR(i, n) s += fabs(x[i*n + j]);
			if (s != 0) {
				FOR(i, n) x[i*n + j] /= s;
			}
		}
		diff = 0;
		FOR(k, dim) diff = max(diff, fabs(y[k]-x[k]));
		if (diff < eps) break;
		if (++nIter > 1000) break;
	}
	delete[] y;
}

void displayMat(double* x, int n1, int n2) {
	FOR(i, n1) {
		FOR(j, n2) {
			printf("%.3f ", x[i*n2 + j]);
		}
		printf("\n");
	}
}

void discritize(const double* X, int n1, int n2, double* Y) {
	double** matrix = array_to_matrix(X, n1, n2);
	hungarian_problem_t hunger;
	hungarian_init(&hunger, matrix, n1, n2, HUNGARIAN_MODE_MAXIMIZE_UTIL) ;
	hungarian_solve(&hunger);
	FOR(i, n1) {
		FOR(j, n2) Y[i*n2+j] = (hunger.assignment[i][j] > 0) ?1: 0;
	}
	hungarian_free(&hunger);
	if (matrix != NULL) {
		for(int i = 0; i < n1; ++i) {
			if (matrix[i] != NULL) {
				free(matrix[i]);
			}
		}
		free(matrix);
	}
}

double computeAcc(double* Xout, const double* gtruth, int n1, int n2, int mode) {
	double acc = 0;
	if (mode == 0) {
		int* match = new int[n1];
		for (int i = 0; i < n1; ++i) {
			match[i] = 0;
			for (int j = 0; j < n2; ++j) {
				if (Xout[i*n2 + j] > Xout[i*n2 + match[i]]) {
					match[i] = j;
				}
			}
			acc += gtruth[i*n2 + match[i]];
		}
		delete[] match;
	} else {
		double** matrix = array_to_matrix(Xout, n1, n2);
		hungarian_problem_t hunger;
		hungarian_init(&hunger, matrix, n1, n2, HUNGARIAN_MODE_MAXIMIZE_UTIL) ;
		hungarian_solve(&hunger);
		FOR(i, n1) {
			FOR (j, n2) {
				acc += (hunger.assignment[i][j] && gtruth[i*n2 + j]) ?1 :0;
			}
		}
		hungarian_free(&hunger);
	}

	double t = 0;
	FOR(i, n1) FOR(j, n2) t += gtruth[i*n2 + j];
	acc = 100 * acc / t;
	return acc;
}
void readArrayFromFile(const char* filename, double* &a) {
	ifstream fin(filename);
	if (!fin.is_open()) {
            cout << "cannot open file: " << filename << endl;
            return;
      }
	double x;
	vector<double> vec;
	while (fin >> x) {
		vec.push_back(x);
	}
	FOR(i, vec.size()) a[i] = vec[i];	
	fin.close();
}

int getRandomTuples(int MAX_TUPLE, int nPoint, int order, int* &tuples, int& nTuple) {
      srand(time(NULL));

      map<vector<int>, bool> ma;
      map<vector<int>, bool>::iterator it;

      for(int t = 0; t < MAX_TUPLE; t++) {
            vector<int> v;
            for(int i = 0; i < order; ++i) {
                  int x = rand() % nPoint;
                  while (find(v.begin(), v.end(), x) != v.end()) {
                        x = rand() % nPoint;
                  }
                  v.push_back(x);
            }
            sort(v.begin(), v.end());
            ma[v] = true;
      }
      
      nTuple = ma.size();
      tuples = new int[nTuple * order];
      int i;
      for(it = ma.begin(), i = 0; it != ma.end(); ++it, ++i) {
            for(int j = 0; j < order; ++j) {
                  tuples[i * order + j] = it->first[j];
                  assert(tuples[i * order + j] >= 0);
                  assert(tuples[i * order + j] < nPoint);
            }
      }

      return nTuple;
}

void ind2sub(double idx, int n, int &x, int &y, int&z) {
      int p = (int)idx;
      z = p % n;
      p = (p / n);
      y = p % n;
      x = p / n;
}

void save_result(double* X, int n1, int n2, double* error, double* score, int nIter, string filename) {
      FILE* f = fopen(filename.c_str(), "w");
      if (f == NULL) {
            cout << "Cannot open file " << filename << endl;
            return;
      }
      fprintf(f, "%d %d\n", n1, n2);
      for (int i = 0; i < n1; ++i) {
            for (int j = 0; j < n2; ++j) {
                  fprintf(f, "%lf ", X[i*n2 + j]);
            }
            fprintf(f, "\n");
      }

      fprintf(f, "\n%d\n", nIter);
      for (int i = 0; i < nIter; ++i) {
            fprintf(f, "%lf ", error[i]);
      }
      fprintf(f, "\n");
      for (int i = 0; i < nIter; ++i) {
            fprintf(f, "%lf ", score[i]);
      }
      fclose(f);
}

void save_array(double* acc, int n, string filename) {
      FILE* f = fopen(filename.c_str(), "w");
      for (int i = 0; i < n; ++i) {
            fprintf(f, "%lf\n", acc[i]);
      }
      fclose(f);
}

#endif
