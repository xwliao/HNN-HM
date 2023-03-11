import numpy as np

from .TensorMatchingCore import wrapTensorMatching


def tensor_matching(problem, max_iter=100, sparse=True, stoc='smart'):
  """
  stoc:
      "default": the matrix will be normalized by its Frobenius norm in each iteration
      "single" : each row of the matrix will be normalized
      "doubly" : Sinkhorn method -- normalize eatch row then each column multiple times
      "smart"  : if nP1 = nP2, then use 'doubly';
                 else if nP1 < nP2, then use "single";
                 otherwise reject to work (i.e. nP2 > nP1 is not recommended).

  output:
    X: nP1 x nP2 (Note: this different from `wrapTensorMatching` which output a nP2 x nP1 matrix)
  """
  assert stoc in ['default', 'single', 'doubly', 'smart']

  nP1 = problem["nP1"]
  nP2 = problem["nP2"]

  if stoc == 'smart':
    assert nP1 <= nP2, "#P1 > #P2 is not recommended"
    if nP1 < nP2:
      stoc = 'single'
    else:
      stoc = 'doubly'

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

  if stoc == 'default':
    stoc = 0
  elif stoc == 'single':
    stoc = 1
  elif stoc == 'doubly':
    stoc = 2

  X = np.ones((nP2, nP1), dtype=np.float32) / nP2
  Xout, score = wrapTensorMatching(
      X,
      indH1, valH1,
      indH2, valH2,
      indH3, valH3,
      max_iter, sparse, stoc)

  # N2 x N1 => N1 x N2
  # TODO: directly output (N1, N2) matrix
  Xout = Xout.T

  return Xout
