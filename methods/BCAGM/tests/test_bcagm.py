import random
import itertools
import numpy as np

from BCAGM.bcagm import bcagm
from BCAGM.bcagm import adapt_bcagm
from BCAGM.bcagm import bcagm_mp
from BCAGM.bcagm import adapt_bcagm_mp
from BCAGM.bcagm import bcagm_ipfp
from BCAGM.bcagm import adapt_bcagm_ipfp
from BCAGM.bcagm import bcagm3
from BCAGM.bcagm import adapt_bcagm3
from BCAGM.bcagm import bcagm3_mp
from BCAGM.bcagm import adapt_bcagm3_mp
from BCAGM.bcagm import bcagm3_ipfp
from BCAGM.bcagm import adapt_bcagm3_ipfp


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


def _test_helper(func, nP1, nP2):
  set_random_seed(seed=1024)
  problem = generate_problem(nP1, nP2)

  Xout1 = func(problem)
  Xout2 = func(problem)
  assert Xout1.shape == (nP1, nP2)
  np.testing.assert_allclose(Xout1, Xout2)


def test_bcagm():
  _test_helper(bcagm, 20, 20)
  _test_helper(bcagm, 10, 20)


def test_adapt_bcagm():
  _test_helper(adapt_bcagm, 20, 20)
  _test_helper(adapt_bcagm, 10, 20)


def test_bcagm_mp():
  _test_helper(bcagm_mp, 20, 20)
  _test_helper(bcagm_mp, 10, 20)


def test_adapt_bcagm_mp():
  _test_helper(adapt_bcagm_mp, 20, 20)
  _test_helper(adapt_bcagm_mp, 10, 20)


def test_bcagm_ipfp():
  _test_helper(bcagm_ipfp, 20, 20)
  _test_helper(bcagm_ipfp, 10, 20)


def test_adapt_bcagm_ipfp():
  _test_helper(adapt_bcagm_ipfp, 20, 20)
  _test_helper(adapt_bcagm_ipfp, 10, 20)


def test_bcagm3():
  _test_helper(bcagm3, 20, 20)
  _test_helper(bcagm3, 10, 20)


def test_adapt_bcagm3():
  _test_helper(adapt_bcagm3, 20, 20)
  _test_helper(adapt_bcagm3, 10, 20)


def test_bcagm3_mp():
  _test_helper(bcagm3_mp, 20, 20)
  _test_helper(bcagm3_mp, 10, 20)


def test_adapt_bcagm3_mp():
  _test_helper(adapt_bcagm3_mp, 20, 20)
  _test_helper(adapt_bcagm3_mp, 10, 20)


def test_bcagm3_ipfp():
  _test_helper(bcagm3_ipfp, 20, 20)
  _test_helper(bcagm3_ipfp, 10, 20)


def test_adapt_bcagm3_ipfp():
  _test_helper(adapt_bcagm3_ipfp, 20, 20)
  _test_helper(adapt_bcagm3_ipfp, 10, 20)


def main():
  test_bcagm()
  test_adapt_bcagm()
  test_bcagm_mp()
  test_adapt_bcagm_mp()
  test_bcagm_ipfp()
  test_adapt_bcagm_ipfp()
  test_bcagm3()
  test_adapt_bcagm3()
  test_bcagm3_mp()
  test_adapt_bcagm3_mp()
  test_bcagm3_ipfp()
  test_adapt_bcagm3_ipfp()


if __name__ == '__main__':
  main()
