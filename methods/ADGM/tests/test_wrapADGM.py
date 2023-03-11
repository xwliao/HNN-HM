import random
import numpy as np

from ADGM import ADGMCore
from ADGM.ADGMCore import wrapADGM3rdOrder
from ADGM.ADGMCore import wrapADGM3rdOrderSymmetric


def set_random_seed(seed):
  random.seed(seed)
  np.random.seed(seed)


def test_wrapADGM3rdOrder():
  set_random_seed(seed=1024)

  nP1 = 10
  nP2 = 20

  X = np.ones((nP2, nP1), dtype=np.float32) / nP2

  indH1 = np.random.choice(range(X.size), (10, 1))
  valH1 = np.random.rand(10, 1)

  indH2 = np.random.choice(range(X.size), (20, 2))
  valH2 = np.random.rand(20, 1)

  indH3 = np.random.choice(range(X.size), (30, 3))
  valH3 = np.random.rand(30, 1)

  rho = nP1 * nP2 / 1000
  eta = 2.0
  n_rhos_max = 20
  rhos = rho * np.power(eta, range(n_rhos_max))

  max_iter = 5000
  verb = False
  restart = False
  iter1 = 200
  iter2 = 50

  variant = ADGMCore.VARIANT_1
  Xout, residuals, rho = wrapADGM3rdOrder(
      X,
      indH1, -valH1,
      indH2, -valH2,
      indH3, -valH3,
      rhos, max_iter,
      verb, restart,
      iter1, iter2,
      variant
  )
  assert Xout.shape == X.shape

  variant = ADGMCore.VARIANT_2
  Xout, residuals, rho = wrapADGM3rdOrder(
      X,
      indH1, -valH1,
      indH2, -valH2,
      indH3, -valH3,
      rhos, max_iter,
      verb, restart,
      iter1, iter2,
      variant
  )
  assert Xout.shape == X.shape


def test_wrapADGM3rdOrderSymmetric():
  set_random_seed(seed=1024)

  nP1 = 10
  nP2 = 20

  X = np.ones((nP2, nP1), dtype=np.float32) / nP2

  indH1 = np.random.choice(range(X.size), (10, 1))
  valH1 = np.random.rand(10, 1)

  indH2 = np.array([[0, 1], [1, 2], [2, 3],
                    [1, 0], [2, 1], [3, 2]], dtype=np.int32)
  valH2 = np.array([[0.1], [0.2], [0.3]] * 2, dtype=np.float64)

  indH3 = np.array([[0, 1, 2], [1, 2, 3], [2, 3, 4],
                    [0, 2, 1], [1, 3, 2], [2, 4, 3],
                    [1, 0, 2], [2, 1, 3], [3, 2, 4],
                    [1, 2, 0], [2, 3, 1], [3, 4, 2],
                    [2, 0, 1], [3, 1, 2], [4, 2, 3],
                    [2, 1, 0], [3, 2, 1], [4, 3, 2]], dtype=np.int32)
  valH3 = np.array([[0.1], [0.2], [0.3]] * 6, dtype=np.float64)

  rho = nP1 * nP2 / 1000
  eta = 2.0
  n_rhos_max = 20
  rhos = rho * np.power(eta, range(n_rhos_max))

  max_iter = 5000
  verb = False
  restart = False
  iter1 = 200
  iter2 = 50

  variant = ADGMCore.VARIANT_1
  Xout, residuals, rho = wrapADGM3rdOrderSymmetric(
      X,
      indH1, -valH1,
      indH2, -valH2,
      indH3, -valH3,
      rhos, max_iter,
      verb, restart,
      iter1, iter2,
      variant
  )
  assert Xout.shape == X.shape

  variant = ADGMCore.VARIANT_2
  Xout, residuals, rho = wrapADGM3rdOrderSymmetric(
      X,
      indH1, -valH1,
      indH2, -valH2,
      indH3, -valH3,
      rhos, max_iter,
      verb, restart,
      iter1, iter2,
      variant
  )
  assert Xout.shape == X.shape


def main():
  test_wrapADGM3rdOrder()
  test_wrapADGM3rdOrderSymmetric()


if __name__ == '__main__':
  main()
