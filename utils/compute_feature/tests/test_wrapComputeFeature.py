import numpy as np

from compute_feature.ComputeFeatureCore import wrapComputeFeature
from compute_feature.ComputeFeatureCore import wrapComputeFeatureSimple
from compute_feature.ComputeFeatureCore import wrapComputeFeatureSimple2


def test_wrapComputeFeature():
  nP1 = 4
  P1 = np.random.rand(nP1, 2)

  nP2 = 5
  P2 = np.random.rand(nP2, 2)

  nT1 = 2
  T1 = np.array([[0, 1, 2], [1, 2, 0]], dtype=np.int32)

  feat1, feat2 = wrapComputeFeature(P1, P2, T1)

  assert feat1.shape == (nT1, 3)
  assert feat2.shape == (nP2**3, 3)


def _test_wrapComputeFeatureSimple_helper(P, T):
  feat2 = wrapComputeFeatureSimple2(P, T)
  assert feat2.shape == (T.shape[0], 3)

  for t, f2 in zip(T, feat2):
    f = wrapComputeFeatureSimple(P, t[0], t[1], t[2])
    np.testing.assert_allclose(f, f2)


def test_wrapComputeFeatureSimple():
  P = np.array([[0, 0], [3, 0], [0, 4]], dtype=np.float32)

  T = np.array([[0, 0, 2]], dtype=np.int32)
  _test_wrapComputeFeatureSimple_helper(P, T)

  T = np.array([[0, 1, 2]], dtype=np.int32)
  _test_wrapComputeFeatureSimple_helper(P, T)

  T = np.array([[0, 1, 2], [1, 2, 0]], dtype=np.int32)
  _test_wrapComputeFeatureSimple_helper(P, T)


def main():
  test_wrapComputeFeature()
  test_wrapComputeFeatureSimple()


if __name__ == '__main__':
  main()
