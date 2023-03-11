import numpy as np
from utils.create_tensor import annquery


def test_annquery():
  feat1 = np.loadtxt('test_data/feat1.txt').T  # nT x 3
  feat2 = np.loadtxt('test_data/feat2.txt').T  # (nP2**3) x 3

  indices_target = np.loadtxt('test_data/inds.txt').T - 1  # nT x nNN
  distances_target = np.loadtxt('test_data/dists.txt').T   # nT x nNN

  nNN = indices_target.shape[1]

  indices, distances = annquery(feat2, feat1, nNN)
  np.testing.assert_allclose(indices, indices_target, rtol=1e-4, atol=0)
  np.testing.assert_allclose(distances, distances_target, rtol=1e-4, atol=0)


def test_annquery2():
  feat1 = np.loadtxt('test_data/feat1.txt').T  # nT x 3
  feat2 = np.loadtxt('test_data/feat2.txt').T  # (nP2**3) x 3

  nNN = min(feat1.shape[0], feat2.shape[0])

  indices, distances = annquery(feat2, feat1, nNN)

  F1 = np.tile(feat1, (1, nNN)).reshape((-1, feat1.shape[1]))
  F2 = feat2[indices.flat]
  distances_target = np.sqrt(np.sum((F1 - F2)**2, axis=-1)).reshape(-1, nNN)

  np.testing.assert_allclose(distances, distances_target, rtol=1e-4, atol=0)


if __name__ == '__main__':
  test_annquery()
  test_annquery2()
