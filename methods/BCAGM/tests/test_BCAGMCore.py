import random
import itertools
import numpy as np

from BCAGM.BCAGMCore import wrapBCAGMMatching
from BCAGM.BCAGMCore import wrapBCAGM_QUADMatching
from BCAGM.BCAGMCore import wrapBCAGM3Matching
from BCAGM.BCAGMCore import wrapBCAGM3_QUADMatching


def set_random_seed(seed):
  random.seed(seed)
  np.random.seed(seed)


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


def _test_helper(func, nP1, nP2, multiples, *args):
  set_random_seed(seed=1024)

  indH1 = np.zeros((0, 1))
  valH1 = np.zeros((0, 1))

  indH2 = np.zeros((0, 2))
  valH2 = np.zeros((0, 1))

  indH3, valH3 = _generate_H3(nP1, nP2)
  indH3, valH3 = _undirected_to_fully_directed(indH3, valH3)

  X0 = np.ones([multiples * nP2 * nP1, 1], dtype=np.float32)

  Xout, objs, nIter = func(indH1.T, valH1,
                           indH2.T, valH2,
                           indH3.T, valH3,
                           nP1, nP2,
                           X0,
                           *args)

  assert Xout.shape == (nP2, nP1)


def test_wrapBCAGMMatching():
  method = wrapBCAGMMatching
  multiples = 4
  max_iter = 50

  adapt = 0
  _test_helper(method, 20, 20, multiples, max_iter, adapt)
  _test_helper(method, 10, 20, multiples, max_iter, adapt)

  adapt = 1
  _test_helper(method, 20, 20, multiples, max_iter, adapt)
  _test_helper(method, 10, 20, multiples, max_iter, adapt)


def test_wrapBCAGM_QUADMatching():
  method = wrapBCAGM_QUADMatching
  multiples = 2

  subroutine = 1
  adapt = 0
  _test_helper(method, 20, 20, multiples, subroutine, adapt)
  _test_helper(method, 10, 20, multiples, subroutine, adapt)

  subroutine = 1
  adapt = 1
  _test_helper(method, 20, 20, multiples, subroutine, adapt)
  _test_helper(method, 10, 20, multiples, subroutine, adapt)

  subroutine = 2
  adapt = 0
  _test_helper(method, 20, 20, multiples, subroutine, adapt)
  _test_helper(method, 10, 20, multiples, subroutine, adapt)

  subroutine = 2
  adapt = 1
  _test_helper(method, 20, 20, multiples, subroutine, adapt)
  _test_helper(method, 10, 20, multiples, subroutine, adapt)


def test_wrapBCAGM3Matching():
  method = wrapBCAGM3Matching
  multiples = 3
  max_iter = 50

  adapt = 0
  _test_helper(method, 20, 20, multiples, max_iter, adapt)
  _test_helper(method, 10, 20, multiples, max_iter, adapt)

  adapt = 1
  _test_helper(method, 20, 20, multiples, max_iter, adapt)
  _test_helper(method, 10, 20, multiples, max_iter, adapt)


def test_wrapBCAGM3_QUADMatching():
  method = wrapBCAGM3_QUADMatching
  multiples = 2

  subroutine = 1
  adapt = 0
  _test_helper(method, 20, 20, multiples, subroutine, adapt)
  _test_helper(method, 10, 20, multiples, subroutine, adapt)

  subroutine = 1
  adapt = 1
  _test_helper(method, 20, 20, multiples, subroutine, adapt)
  _test_helper(method, 10, 20, multiples, subroutine, adapt)

  subroutine = 2
  adapt = 0
  _test_helper(method, 20, 20, multiples, subroutine, adapt)
  _test_helper(method, 10, 20, multiples, subroutine, adapt)

  subroutine = 2
  adapt = 1
  _test_helper(method, 20, 20, multiples, subroutine, adapt)
  _test_helper(method, 10, 20, multiples, subroutine, adapt)


def main():
  test_wrapBCAGMMatching()
  test_wrapBCAGM_QUADMatching()
  test_wrapBCAGM3Matching()
  test_wrapBCAGM3_QUADMatching()


if __name__ == '__main__':
  main()
