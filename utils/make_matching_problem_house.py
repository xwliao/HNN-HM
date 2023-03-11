import numpy as np

from . import house
from .create_problem import create_problem
from .create_problem import convert_to_directed  # noqa


def _generate_point_sets(index1, index2, nP1, nP2, shuffle_points, shuffle_assignment, rng=None):
  rng = np.random.default_rng(rng)

  assert 0 <= index1 < house.DATASET_SIZE
  assert 0 <= index2 < house.DATASET_SIZE

  assert 0 < nP1 <= house.NUM_POINTS
  assert 0 < nP2 <= house.NUM_POINTS

  assert nP1 <= nP2

  if shuffle_points:
      inds = rng.permutation(house.NUM_POINTS)
  else:
      inds = np.arange(house.NUM_POINTS)

  P1 = house.get_points(index1)[inds][:nP1]
  P2 = house.get_points(index2)[inds][:nP2]

  # permute graph sequence (prevent accidental good solution)
  #   `trueMatch` start from 0 (different from the original matlab code whic start from 1)
  #   trueMatch[i] = j means P1[:, i] matches P2[:, j]
  if shuffle_assignment:
    trueMatch = rng.permutation(nP2)
    P2[trueMatch, :] = P2.copy()  # NOTE: Don't write P2[trueMatch, :] = P2, which is buggy.
    trueMatch = trueMatch[:nP1]
  else:
    trueMatch = np.arange(nP1)

  assignmentMatrix = np.zeros((nP1, nP2), np.bool)
  for p1, p2 in enumerate(trueMatch):
    assignmentMatrix[p1, p2] = True

  return P1, P2, assignmentMatrix


def make_matching_problem(index1, index2, nP1, nP2, shuffle_points, shuffle_assignment, scale, rng=None):
  """
  Output:
    problem: dict
    {
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

  P1, P2, assignmentMatrix = _generate_point_sets(index1, index2, nP1, nP2,
                                                  shuffle_points=shuffle_points,
                                                  shuffle_assignment=shuffle_assignment,
                                                  rng=rng)

  problem = create_problem(P1, P2,
                           assignmentMatrix,
                           scale=scale,
                           order=3,
                           rng=rng)

  return problem
