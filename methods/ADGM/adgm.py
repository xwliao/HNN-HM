import numpy as np

from .ADGMCore import VARIANT_1
from .ADGMCore import VARIANT_2

from .ADGMCore import wrapADGM3rdOrder
from .ADGMCore import wrapADGM3rdOrderSymmetric


def _adgm_helper(func, problem, variant):
  """
  output:
    X: nP1 x nP2 (Note: this different from `wrapADGM*` which output a nP2 x nP1 matrix)
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

  rho = nP1 * nP2 / 1000
  eta = 2.0
  n_rhos_max = 20
  rhos = rho * np.power(eta, range(n_rhos_max))

  max_iter = 5000
  verb = False
  restart = False
  iter1 = 200
  iter2 = 50

  X = np.ones((nP2, nP1), dtype=np.float32) / nP2
  Xout, _, _ = func(
      X,
      indH1, -valH1,
      indH2, -valH2,
      indH3, -valH3,
      rhos, max_iter,
      verb, restart,
      iter1, iter2,
      variant
  )

  # N2 x N1 => N1 x N2
  # TODO: directly output (N1, N2) matrix
  Xout = Xout.T

  return Xout


def adgm1(problem):
  func = wrapADGM3rdOrder
  variant = VARIANT_1
  X = _adgm_helper(func, problem, variant=variant)
  return X


def adgm2(problem):
  func = wrapADGM3rdOrder
  variant = VARIANT_2
  X = _adgm_helper(func, problem, variant=variant)
  return X


def adgm1_symmetric(problem):
  func = wrapADGM3rdOrderSymmetric
  variant = VARIANT_1
  X = _adgm_helper(func, problem, variant=variant)
  return X


def adgm2_symmetric(problem):
  func = wrapADGM3rdOrderSymmetric
  variant = VARIANT_2
  X = _adgm_helper(func, problem, variant=variant)
  return X
