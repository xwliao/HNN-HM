import numpy as np

from .RRWHMCore import wrapRRWHM


def RRWHM(problem, max_iter=300, c=0.2):
  """
  Output:
    X: nP1 x nP2
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

  # Note that whatever the scale is, the input X will still be normalized in the C++ code.
  X = np.ones((nP2, nP1), dtype=np.float32) / (nP2 * nP1)

  Xout = wrapRRWHM(
      X,
      indH1, valH1,
      indH2, valH2,
      indH3, valH3,
      max_iter, c)

  # N2 x N1 => N1 x N2
  # TODO: directly output (N1, N2) matrix
  Xout = Xout.T

  return Xout
