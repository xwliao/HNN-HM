import random
import numpy as np

from TM.TensorMatchingCore import wrapTensorMatching


def set_random_seed(seed):
  random.seed(seed)
  np.random.seed(seed)


def test_wrapTensorMatching():
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

  nIter = 100
  sparse = True

  for stoc in [0, 1, 2]:
    Xout, score = wrapTensorMatching(
        X,
        indH1, valH1,
        indH2, valH2,
        indH3, valH3,
        nIter, sparse, stoc)

    assert Xout.shape == X.shape


def main():
  test_wrapTensorMatching()


if __name__ == '__main__':
  main()
