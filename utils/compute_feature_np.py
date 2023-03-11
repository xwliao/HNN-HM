import numpy as np


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

  T1 = np.asarray(T1, dtype=np.int32)
  feat1 = compute_feature_simple(P1, T1, dtype)

  n2 = P2.shape[0]
  T2 = np.mgrid[0:n2, 0:n2, 0:n2].reshape(3, -1).astype(np.int32).T
  feat2 = compute_feature_simple(P2, T2, dtype)

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

  # TODO: Convert to double?
  # P = np.asarray(P, dtype=np.float64)
  P = np.asarray(P)

  T = np.asarray(T, dtype=np.int32)

  m = T.shape[0]
  feat = np.zeros((m, 3), dtype=dtype)

  t1 = T[:, 0]
  t2 = T[:, 1]
  t3 = T[:, 2]

  t = np.logical_or(t1 == t2, t1 == t3)
  t = np.logical_or(t, t2 == t3)

  feat[t, :] = -10.0

  if np.all(t):
    return feat

  T2 = T[~t, :]
  m2 = T2.shape[0]

  Pmat = np.eye(3, 3) - np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
  Y = P.T[:, T2.flat].reshape(2, m2, 3)
  Y = np.matmul(Y.reshape(2 * m2, 3), Pmat).reshape(2, m2, 3)
  Ynorm = np.linalg.norm(Y, axis=0, keepdims=True)
  Ynorm[Ynorm <= 0] = np.finfo(Y.dtype).eps
  DX = Y / Ynorm

  x = DX[0, :, :].reshape(m2, 3)
  y = DX[1, :, :].reshape(m2, 3)

  feat[~t, :] = -np.cross(x, y)

  return feat


if __name__ == '__main__':
  P = np.array([[0, 0], [3, 0], [0, 4]], dtype=np.float32)

  T = np.array([[0, 0, 2]], dtype=np.int32)
  feat = compute_feature_simple(P, T)
  print(feat)

  T = np.array([[0, 1, 2]], dtype=np.int32)
  feat = compute_feature_simple(P, T)
  print(feat)

  T = np.array([[0, 1, 2], [1, 2, 0]], dtype=np.int32)
  feat = compute_feature_simple(P, T)
  print(feat)
