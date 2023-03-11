import numpy as np
from .hungarian import hungarian


def greedy_mapping(X):
  X = X.copy()
  Y = np.zeros_like(X)
  for _ in range(np.min(X.shape)):
    flat_idx = np.argmax(X, axis=None)
    i, j = np.unravel_index(flat_idx, X.shape)
    Y[i, j] = 1
    X[i, :] = 0
    X[:, j] = 0
  return Y


def getMatchScore(indH, valH, X):
  """
  H: 3rd order potential
  X: Assignment matrix
  """
  T = X.flat[indH]
  return np.einsum('i,i,i,i->', valH.flat, T[:, 0], T[:, 1], T[:, 2])


def check_assignment_tensor(X):
  # Check binary
  # TODO: Is using `!=` to check float numbers OK?
  if np.any(np.logical_and(X != 0, X != 1)):
    return False
  # TODO: What if all sum_along_axis < 1? Should at least one is equal to one?
  for axis in range(X.ndim):
    sum_along_axis = np.sum(X, axis=axis)
    if np.any(sum_along_axis > 1):
      return False
  return True


def evaluate(problem, X, lap_solver='hungarian'):
  """
  problem should have these keys: assignmentMatrix, indH3, valH3
  """
  indH = problem["indH3"]
  valH = problem["valH3"]
  Xgt = problem["assignmentMatrix"].astype(np.float32)

  assert lap_solver in ['hungarian', 'greedy', 'identity']
  assert X.shape == Xgt.shape

  if lap_solver == 'hungarian':
    Xbin = hungarian(X)
  elif lap_solver == 'greedy':
    Xbin = greedy_mapping(X)
  else:
    Xbin = X.copy()
    assert check_assignment_tensor(Xbin)
  accuracy = np.sum(Xbin * Xgt, dtype=np.float32) / np.sum(Xgt)
  match_score = getMatchScore(indH, valH, Xbin)
  return accuracy, match_score
