import numpy as np

from .create_problem import create_problem
from .create_problem import convert_to_directed  # noqa


def get_rotation_matrix(theta):
  c = np.cos(theta)
  s = np.sin(theta)
  return np.array([[c, -s], [s, c]])


def _generate_point_sets(settings, rng=None):
  rng = np.random.default_rng(rng)

  nInlier = settings["nInlier"]
  nOutlier = settings["nOutlier"]
  deformation = settings["deformation"]
  typeDistribution = settings["typeDistribution"]  # normal / uniform
  transScale = settings["transScale"]    # scale change
  transRotate = settings["transRotate"]  # rotation change
  bPermute = settings["bPermute"]

  assert "bOutBoth" not in settings  # TODO: To be removed

  nP1 = nInlier
  nP2 = nInlier + nOutlier

  assert typeDistribution in ['normal', 'uniform']

  # Points in Domain 1
  if typeDistribution == 'normal':
    P1 = rng.standard_normal(size=(nP1, 2))
    Pout = rng.standard_normal(size=(nOutlier, 2))
  elif typeDistribution == 'uniform':
    P1 = rng.random(size=(nP1, 2))
    Pout = rng.random(size=(nOutlier, 2))

  # point transformation matrix
  Mrot = get_rotation_matrix(transRotate)

  # Poinst in Domain 2
  P2 = transScale * np.matmul(P1, Mrot.T) + deformation * rng.standard_normal(size=P1.shape)
  # P2 = np.r_[P2, Pout]
  P2 = np.concatenate((P2, Pout), axis=0)

  # permute graph sequence (prevent accidental good solution)
  #   `trueMatch` start from 0 (different from the original matlab code whic start from 1)
  #   trueMatch[i] = j means P1[:, i] matches P2[:, j]
  if bPermute:
    trueMatch = rng.permutation(nP2)
    P2[trueMatch, :] = P2.copy()  # NOTE: Don't write P2[trueMatch, :] = P2, which is buggy.
    trueMatch = trueMatch[:nP1]
  else:
    trueMatch = np.arange(nP1)

  assignmentMatrix = np.zeros((nP1, nP2), np.bool)
  for p1, p2 in enumerate(trueMatch):
    assignmentMatrix[p1, p2] = True

  return P1, P2, assignmentMatrix


def make_matching_problem(settings, rng=None):
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

  scale = settings["scale"]

  P1, P2, assignmentMatrix = _generate_point_sets(settings, rng=rng)

  problem = create_problem(P1, P2,
                           assignmentMatrix,
                           scale=scale,
                           order=3,
                           rng=rng)

  return problem


if __name__ == '__main__':
  settings = {
      "nInlier": 20,
      "nOutlier": 0,
      "deformation": 0.025,
      "typeDistribution": 'normal',  # normal / uniform
      "transScale": 1.0,   # scale change
      "transRotate": 0.0,  # rotation change
      "scale": 0.2,
      "bPermute": True
  }
  problem = make_matching_problem(settings)
