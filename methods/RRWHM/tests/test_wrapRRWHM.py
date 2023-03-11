import random
import numpy as np

from RRWHM.RRWHMCore import wrapRRWHM


def set_random_seed(seed):
  random.seed(seed)
  np.random.seed(seed)


def test_wrapRRWHM():
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

  nIter = 300
  c = 0.2

  Xout = wrapRRWHM(
      X,
      indH1, valH1,
      indH2, valH2,
      indH3, valH3,
      nIter, c)

  assert Xout.shape == X.shape


def main():
  test_wrapRRWHM()


if __name__ == '__main__':
  main()
