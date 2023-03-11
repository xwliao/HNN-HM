import numpy as np

from compute_feature import compute_feature
from compute_feature import compute_feature_simple


def test_compute_feature():
  nP1 = 4
  P1 = np.random.rand(nP1, 2)

  nP2 = 5
  P2 = np.random.rand(nP2, 2)

  nT1 = 2
  T1 = np.array([[0, 1, 2], [1, 2, 0]], dtype=np.int32)

  feat1, feat2 = compute_feature(P1, P2, T1)

  assert feat1.shape == (nT1, 3)
  assert feat2.shape == (nP2**3, 3)


def test_compute_feature_simple():
  P = np.array([[0, 0], [3, 0], [0, 4]], dtype=np.float32)

  T = np.array([[0, 0, 2]], dtype=np.int32)
  feat = compute_feature_simple(P, T)
  assert feat.shape == (1, 3)

  T = np.array([[0, 1, 2]], dtype=np.int32)
  feat = compute_feature_simple(P, T)
  assert feat.shape == (1, 3)

  T = np.array([[0, 1, 2], [1, 2, 0]], dtype=np.int32)
  feat = compute_feature_simple(P, T)
  assert feat.shape == (2, 3)


def main():
  test_compute_feature()
  test_compute_feature_simple()


if __name__ == '__main__':
  main()
