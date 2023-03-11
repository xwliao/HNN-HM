import numpy as np

from . import willow
from .create_problem import create_problem
from .create_problem import convert_to_directed  # noqa


def get_cid(mat_file):
  category = willow.get_category(mat_file)
  cid = willow.CATEGORIES.index(category)
  return cid


def _generate_point_sets(mat_file1, mat_file2, n_inlier, n_outlier, shuffle, rng=None):
  rng = np.random.default_rng(rng)

  n1 = n_inlier
  n2 = n_inlier + n_outlier
  pts1 = willow.get_points(mat_file1)[:n1]
  pts2 = willow.get_points(mat_file2)[:n1]
  if n_outlier > 0:
    outliers = willow.generate_points(mat_file2, npts=n_outlier, rng=rng)
    pts2 = np.r_[pts2, outliers]

  if shuffle:
    perm1 = rng.permutation(n1)
    perm2 = rng.permutation(n2)
  else:
    perm1 = np.arange(n1)
    perm2 = np.arange(n2)

  pts1 = pts1[perm1]
  pts2 = pts2[perm2]

  assignmentMatrix = np.eye(n1, n2, dtype=np.bool)
  assignmentMatrix = assignmentMatrix[perm1]
  assignmentMatrix = assignmentMatrix[:, perm2]

  return pts1, pts2, assignmentMatrix


def make_matching_problem(mat_file1, mat_file2, n1, n2, shuffle, scale, rng=None):
  """
  Output:
    problem: dict
    {
      cid:   int
      nP1:   int
      nP2:   int
      P1:    float64, (nP1, 2)
      P2:    float64, (nP2, 2)
      indH1: int32,   (N1, 1) array or None
      valH1: float64, (N1,)   array or None
      indH2: int32,   (N2, 2) array or None
      valH2: float64, (N2,)   array or None
      indH3: int32,   (N3, 3) array or None
      valH3: float64, (N3,)   array or None
      assignmentMatrix: bool, (nP1, nP2)
    }
  """
  rng = np.random.default_rng(rng)

  assert 1 <= n1 <= willow.NUM_POINTS
  assert n1 <= n2

  cid1 = get_cid(mat_file1)
  cid2 = get_cid(mat_file2)
  assert cid1 == cid2
  cid = cid1

  P1, P2, assignmentMatrix = _generate_point_sets(mat_file1, mat_file2,
                                                  n_inlier=n1, n_outlier=n2 - n1,
                                                  shuffle=shuffle,
                                                  rng=rng)

  problem = create_problem(P1, P2,
                           assignmentMatrix,
                           scale=scale,
                           cid=cid,
                           order=3,
                           rng=rng)

  return problem
