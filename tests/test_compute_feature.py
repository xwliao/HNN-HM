import numpy as np
from utils.compute_feature import compute_feature, compute_feature_simple


def test_compute_feature():

  P1 = np.loadtxt('test_data/P1.txt').T  # nP1 x 2
  P2 = np.loadtxt('test_data/P2.txt').T  # nP2 x 2

  T1 = np.loadtxt('test_data/T1.txt', dtype=np.int32).T  # nT x 3

  feat1 = np.loadtxt('test_data/feat1.txt').T  # nT x 3
  feat2 = np.loadtxt('test_data/feat2.txt').T  # (nP2**3) x 3

  feat1_compute = compute_feature_simple(P1, T1)
  np.testing.assert_allclose(feat1_compute, feat1, rtol=1e-4, atol=1e-6)

  feat1_compute, feat2_compute = compute_feature(P1, P2, T1)
  np.testing.assert_allclose(feat1_compute, feat1, rtol=1e-4, atol=1e-6)
  np.testing.assert_allclose(feat2_compute, feat2, rtol=1e-4, atol=1e-6)


if __name__ == '__main__':
  test_compute_feature()
