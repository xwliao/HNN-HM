import numpy as np

from .ComputeFeatureCore import wrapComputeFeature
from .ComputeFeatureCore import wrapComputeFeatureSimple2


def compute_feature(P1, P2, T1, dtype=None):
  """
  Input:
  P1: nP1 x 2 (nP1 2d points)
  P2: nP2 x 2 (nP2 2d points), should be the same type as P1
  T1: nT1 x 3 (nT1 pairs of triangle; index begins with 0)
  dtype: Type of the output (default is the same as P1 and P2)

  Output:
  feat1: nT1 x 3 (nT1 features)
  feat2: (nP2^3) x 3 ((nP2^3) features)
  """
  assert P1.shape[1] == 2, 'Shape of P1 should be (nP1, 2)'
  assert P2.shape[1] == 2, 'Shape of P2 should be (nP2, 2)'
  assert T1.shape[1] == 3, 'Shape of T1 should be (nT1, 3)'

  assert P1.dtype == P2.dtype, 'Type of P1 and P2 should be the same!'
  if dtype is None:
    dtype = P1.dtype

  # Convert to double which is used by wrapComputeFeature
  P1 = np.asarray(P1, dtype=np.float64)
  P2 = np.asarray(P2, dtype=np.float64)

  T1 = np.asarray(T1, dtype=np.int32)

  feat1, feat2 = wrapComputeFeature(P1, P2, T1)
  feat1 = feat1.astype(dtype, copy=False)
  feat2 = feat2.astype(dtype, copy=False)

  return feat1, feat2


def compute_feature_simple(P, T, dtype=None):
  """
  Input:
  P: nP x 2 (nP 2d points)
  T: nT x 3 (nT pairs of triangle)
  dtype: Type of the output (default is the same as P)

  Output:
  feat: nT x 3 (m features)
  """
  assert P.shape[1] == 2, 'Shape of P should be (nP, 2)'
  assert T.shape[1] == 3, 'Shape of T should be (nT, 3)'

  if dtype is None:
    dtype = P.dtype

  # Convert to double which is used by wrapComputeFeatureSimple2
  P = np.asarray(P, dtype=np.float64)

  T = np.asarray(T, dtype=np.int32)

  feat = wrapComputeFeatureSimple2(P, T)
  feat = feat.astype(dtype, copy=False)

  return feat
