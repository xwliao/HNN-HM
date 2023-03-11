import random
import itertools
import numpy as np

from BCAGM.bcagm_basic import bcagm
from BCAGM.bcagm_basic import bcagm_quad
from BCAGM.bcagm_basic import bcagm3
from BCAGM.bcagm_basic import bcagm3_quad


def set_random_seed(seed):
  random.seed(seed)
  np.random.seed(seed)


# def generate_problem(nP1, nP2):
#   N = nP2 * nP1
#   indH1 = np.random.choice(range(N), (10, 1))
#   valH1 = np.random.rand(10, 1)

#   indH2 = np.random.choice(range(N), (20, 2))
#   valH2 = np.random.rand(20, 1)

#   indH3 = np.random.choice(range(N), (30, 3))
#   valH3 = np.random.rand(30, 1)

#   problem = {
#       "nP1": nP1,
#       "nP2": nP2,
#       "indH1": indH1,
#       "valH1": valH1,
#       "indH2": indH2,
#       "valH2": valH2,
#       "indH3": indH3,
#       "valH3": valH3,
#   }

#   return problem


def _generate_H3(nP1, nP2):
  n = min(nP1, nP2)

  a = np.arange(n - 2)
  ind1 = np.c_[a, a + 1, a + 2]

  ind2 = ind1

  if nP1 > nP2:
    ind1 = ind1 + (nP1 - nP2)
  else:
    ind2 = ind2 + (nP2 - nP1)

  indH3 = np.ravel_multi_index([ind2, ind1], (nP2, nP1), order='F')

  valH3 = np.random.rand(indH3.shape[0], 1)

  return indH3, valH3


def _undirected_to_fully_directed(indH, valH):
  order = indH.shape[-1]
  indices = np.asarray(list(itertools.permutations(range(order))))
  indH = indH[:, indices].reshape(-1, order)
  N = len(indices)
  valH = np.repeat(valH, repeats=N, axis=0)
  return indH, valH


def generate_problem(nP1, nP2):
  indH1 = np.zeros((0, 1))
  valH1 = np.zeros((0, 1))

  indH2 = np.zeros((0, 2))
  valH2 = np.zeros((0, 1))

  # TODO
  indH3, valH3 = _generate_H3(nP1, nP2)
  indH3, valH3 = _undirected_to_fully_directed(indH3, valH3)

  problem = {
      "nP1": nP1,
      "nP2": nP2,
      "indH1": indH1,
      "valH1": valH1,
      "indH2": indH2,
      "valH2": valH2,
      "indH3": indH3,
      "valH3": valH3,
  }

  return problem


def _test_helper(func, nP1, nP2, multiples, *args, **kwargs):
  set_random_seed(seed=1024)
  problem = generate_problem(nP1, nP2)
  X0 = np.ones([multiples * nP2 * nP1, 1], dtype=np.float32)
  Xout = func(problem, X0, *args, **kwargs)
  assert Xout.shape == (nP2, nP1)


def test_bcagm():
  _test_helper(bcagm, 20, 20, 4, max_iter=50, adapt=0)
  _test_helper(bcagm, 10, 20, 4, max_iter=50, adapt=0)

  _test_helper(bcagm, 20, 20, 4, max_iter=50, adapt=1)
  _test_helper(bcagm, 10, 20, 4, max_iter=50, adapt=1)


def test_bcagm_quad():
  _test_helper(bcagm_quad, 20, 20, 2, subroutine=1, adapt=0)
  _test_helper(bcagm_quad, 10, 20, 2, subroutine=1, adapt=0)

  _test_helper(bcagm_quad, 20, 20, 2, subroutine=1, adapt=1)
  _test_helper(bcagm_quad, 10, 20, 2, subroutine=1, adapt=1)

  _test_helper(bcagm_quad, 20, 20, 2, subroutine=2, adapt=0)
  _test_helper(bcagm_quad, 10, 20, 2, subroutine=2, adapt=0)

  _test_helper(bcagm_quad, 20, 20, 2, subroutine=2, adapt=1)
  _test_helper(bcagm_quad, 10, 20, 2, subroutine=2, adapt=1)


def test_bcagm3():
  _test_helper(bcagm3, 20, 20, 3, max_iter=50, adapt=0)
  _test_helper(bcagm3, 10, 20, 3, max_iter=50, adapt=0)

  _test_helper(bcagm3, 20, 20, 3, max_iter=50, adapt=1)
  _test_helper(bcagm3, 10, 20, 3, max_iter=50, adapt=1)


def test_bcagm3_quad():
  _test_helper(bcagm3_quad, 20, 20, 2, subroutine=1, adapt=0)
  _test_helper(bcagm3_quad, 10, 20, 2, subroutine=1, adapt=0)

  _test_helper(bcagm3_quad, 20, 20, 2, subroutine=1, adapt=1)
  _test_helper(bcagm3_quad, 10, 20, 2, subroutine=1, adapt=1)

  _test_helper(bcagm3_quad, 20, 20, 2, subroutine=2, adapt=0)
  _test_helper(bcagm3_quad, 10, 20, 2, subroutine=2, adapt=0)

  _test_helper(bcagm3_quad, 20, 20, 2, subroutine=2, adapt=1)
  _test_helper(bcagm3_quad, 10, 20, 2, subroutine=2, adapt=1)


def main():
  test_bcagm()
  test_bcagm_quad()
  test_bcagm3()
  test_bcagm3_quad()


if __name__ == '__main__':
  main()
