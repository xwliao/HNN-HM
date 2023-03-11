import random
import numpy as np

from utils.compute_feature import compute_feature as compute_feature_cpp
from utils.compute_feature import compute_feature_simple as compute_feature_simple_cpp

from utils.compute_feature_np import compute_feature as compute_feature_np
from utils.compute_feature_np import compute_feature_simple as compute_feature_simple_np


def set_random_seed(seed):
  random.seed(seed)
  np.random.seed(seed)


def _test_compute_feature_helper(nP1, nP2, nT1, rtol=1e-7, atol=0):
  P1 = np.random.rand(nP1, 2)
  P2 = np.random.rand(nP2, 2)
  T1 = np.random.randint(nP1, size=(nT1, 3), dtype=np.int32)

  feat1_cpp, feat2_cpp = compute_feature_cpp(P1, P2, T1)
  feat1_np, feat2_np = compute_feature_np(P1, P2, T1)

  assert feat1_cpp.shape == (nT1, 3)
  assert feat2_cpp.shape == (nP2 * nP2 * nP2, 3)

  np.testing.assert_allclose(feat1_np, feat1_cpp, rtol=rtol, atol=atol)
  np.testing.assert_allclose(feat2_np, feat2_cpp, rtol=rtol, atol=atol)


def test_compute_feature():
  set_random_seed(seed=1024)

  _test_compute_feature_helper(nP1=3, nP2=3, nT1=1)
  _test_compute_feature_helper(nP1=20, nP2=20, nT1=50)


def _test_compute_feature_simple_helper(nP, nT, rtol=1e-7, atol=0):
  P = np.random.rand(nP, 2)
  T = np.random.randint(nP, size=(nT, 3), dtype=np.int32)

  feat_cpp = compute_feature_simple_cpp(P, T)
  feat_np = compute_feature_simple_np(P, T)

  assert feat_cpp.shape == (nT, 3)
  np.testing.assert_allclose(feat_np, feat_cpp, rtol=rtol, atol=atol)


def test_compute_feature_simple():
  set_random_seed(seed=1024)

  _test_compute_feature_simple_helper(nP=3, nT=1)
  _test_compute_feature_simple_helper(nP=500, nT=10000)


def test_compute_feature_simple2():
  P = np.array([[0, 0], [0.6, 0], [0.6001, 0.0001]], dtype=np.float64)
  T = np.array([[0, 1, 2]], dtype=np.int32)
  nT = 1

  feat_cpp = compute_feature_simple_cpp(P, T)
  feat_np = compute_feature_simple_np(P, T)

  assert feat_cpp.shape == (nT, 3)
  np.testing.assert_allclose(feat_np, feat_cpp)

  P = P.astype(np.float32)
  feat_cpp = compute_feature_simple_cpp(P, T)
  feat_np = compute_feature_simple_np(P, T)

  assert feat_cpp.shape == (nT, 3)
  np.testing.assert_allclose(feat_np, feat_cpp)


if __name__ == '__main__':
  test_compute_feature()
  test_compute_feature_simple()
