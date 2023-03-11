import numpy as np

from .BCAGMCore import wrapBCAGMMatching
from .BCAGMCore import wrapBCAGM_QUADMatching
from .BCAGMCore import wrapBCAGM3Matching
from .BCAGMCore import wrapBCAGM3_QUADMatching


def bcagm(problem, X0, max_iter, adapt):
  """
  NOTE that the input arrays: problem.indH3 and problem.valH3
  should be symmetric themselves.
  This means they should store all the six permutations of
  each non-zero entries instead of storing only one.

  Output:
    X: nP2 x nP1
  """
  nP1 = problem["nP1"]
  nP2 = problem["nP2"]

  indH1 = problem["indH1"]
  valH1 = problem["valH1"]

  indH2 = problem["indH2"]
  valH2 = problem["valH2"]

  indH3 = problem["indH3"]
  valH3 = problem["valH3"]

  if indH1 is None and valH1 is None:
    indH1 = np.zeros((0, 1))
    valH1 = np.zeros((0, 1))

  indH1 = indH1.astype(np.int32, copy=False)
  valH1 = valH1.astype(np.float64, copy=False)

  if indH2 is None and valH2 is None:
    indH2 = np.zeros((0, 2))
    valH2 = np.zeros((0, 1))

  indH2 = indH2.astype(np.int32, copy=False)
  valH2 = valH2.astype(np.float64, copy=False)

  if indH3 is None and valH3 is None:
    indH3 = np.zeros((0, 3))
    valH3 = np.zeros((0, 1))

  indH3 = indH3.astype(np.int32, copy=False)
  valH3 = valH3.astype(np.float64, copy=False)

  X0 = X0.astype(np.float64, copy=False)

  Xout, objs, nIter = wrapBCAGMMatching(
      indH1.T, valH1,
      indH2.T, valH2,
      indH3.T, valH3,
      nP1, nP2,
      X0,
      max_iter, adapt)

  # return Xout, objs, nIter
  return Xout


def bcagm_quad(problem, X0, subroutine, adapt):
  """
  NOTE that the input arrays: problem.indH3 and problem.valH3
  should be symmetric themselves.
  This means they should store all the six permutations of
  each non-zero entries instead of storing only one.

  Output:
    X: nP2 x nP1
  """
  nP1 = problem["nP1"]
  nP2 = problem["nP2"]

  indH1 = problem["indH1"]
  valH1 = problem["valH1"]

  indH2 = problem["indH2"]
  valH2 = problem["valH2"]

  indH3 = problem["indH3"]
  valH3 = problem["valH3"]

  if indH1 is None and valH1 is None:
    indH1 = np.zeros((0, 1))
    valH1 = np.zeros((0, 1))

  indH1 = indH1.astype(np.int32, copy=False)
  valH1 = valH1.astype(np.float64, copy=False)

  if indH2 is None and valH2 is None:
    indH2 = np.zeros((0, 2))
    valH2 = np.zeros((0, 1))

  indH2 = indH2.astype(np.int32, copy=False)
  valH2 = valH2.astype(np.float64, copy=False)

  if indH3 is None and valH3 is None:
    indH3 = np.zeros((0, 3))
    valH3 = np.zeros((0, 1))

  indH3 = indH3.astype(np.int32, copy=False)
  valH3 = valH3.astype(np.float64, copy=False)

  X0 = X0.astype(np.float64, copy=False)

  Xout, objs, nIter = wrapBCAGM_QUADMatching(
      indH1.T, valH1,
      indH2.T, valH2,
      indH3.T, valH3,
      nP1, nP2,
      X0,
      subroutine, adapt)

  # return Xout, objs, nIter
  return Xout


def bcagm3(problem, X0, max_iter, adapt):
  """Multilinear Solver

  Output:
    X: nP2 x nP1
  """
  nP1 = problem["nP1"]
  nP2 = problem["nP2"]

  indH1 = problem["indH1"]
  valH1 = problem["valH1"]

  indH2 = problem["indH2"]
  valH2 = problem["valH2"]

  indH3 = problem["indH3"]
  valH3 = problem["valH3"]

  if indH1 is None and valH1 is None:
    indH1 = np.zeros((0, 1))
    valH1 = np.zeros((0, 1))

  indH1 = indH1.astype(np.int32, copy=False)
  valH1 = valH1.astype(np.float64, copy=False)

  if indH2 is None and valH2 is None:
    indH2 = np.zeros((0, 2))
    valH2 = np.zeros((0, 1))

  indH2 = indH2.astype(np.int32, copy=False)
  valH2 = valH2.astype(np.float64, copy=False)

  if indH3 is None and valH3 is None:
    indH3 = np.zeros((0, 3))
    valH3 = np.zeros((0, 1))

  indH3 = indH3.astype(np.int32, copy=False)
  valH3 = valH3.astype(np.float64, copy=False)

  X0 = X0.astype(np.float64, copy=False)

  Xout, objs, nIter = wrapBCAGM3Matching(
      indH1.T, valH1,
      indH2.T, valH2,
      indH3.T, valH3,
      nP1, nP2,
      X0,
      max_iter, adapt)

  # return Xout, objs, nIter
  return Xout


def bcagm3_quad(problem, X0, subroutine, adapt):
  """Multilinear Solver

  Output:
    X: nP2 x nP1
  """
  nP1 = problem["nP1"]
  nP2 = problem["nP2"]

  indH1 = problem["indH1"]
  valH1 = problem["valH1"]

  indH2 = problem["indH2"]
  valH2 = problem["valH2"]

  indH3 = problem["indH3"]
  valH3 = problem["valH3"]

  if indH1 is None and valH1 is None:
    indH1 = np.zeros((0, 1))
    valH1 = np.zeros((0, 1))

  indH1 = indH1.astype(np.int32, copy=False)
  valH1 = valH1.astype(np.float64, copy=False)

  if indH2 is None and valH2 is None:
    indH2 = np.zeros((0, 2))
    valH2 = np.zeros((0, 1))

  indH2 = indH2.astype(np.int32, copy=False)
  valH2 = valH2.astype(np.float64, copy=False)

  if indH3 is None and valH3 is None:
    indH3 = np.zeros((0, 3))
    valH3 = np.zeros((0, 1))

  indH3 = indH3.astype(np.int32, copy=False)
  valH3 = valH3.astype(np.float64, copy=False)

  X0 = X0.astype(np.float64, copy=False)

  Xout, objs, nIter = wrapBCAGM3_QUADMatching(
      indH1.T, valH1,
      indH2.T, valH2,
      indH3.T, valH3,
      nP1, nP2,
      X0,
      subroutine, adapt)

  # return Xout, objs, nIter
  return Xout
