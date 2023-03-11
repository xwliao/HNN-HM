import numpy as np
from .ipfp import ipfp_3rd


def ipfp_hm(problem, max_iter=20, max_iter_2nd=50):
  """
  output:
    X: nP1 x nP2
  """
  nP1 = problem["nP1"]
  nP2 = problem["nP2"]

  indH3 = problem["indH3"]
  valH3 = problem["valH3"]

  if indH3 is None and valH3 is None:
    indH3 = np.zeros((0, 3))
    valH3 = np.zeros((0, 1))

  indH3 = indH3.astype(np.int32, copy=False)
  valH3 = valH3.astype(np.float64, copy=False)

  X = np.ones((nP1, nP2), dtype=np.float32) / nP2

  eps = 1e-12
  step_eps = 0.
  internal_dtype = 'float64'
  vec_order = 'C'
  convert_to_directed = True
  Xout = ipfp_3rd(indH3, valH3, X,
                  max_iter=max_iter,
                  max_iter_2nd=max_iter_2nd,
                  eps=eps,
                  step_eps=step_eps,
                  internal_dtype=internal_dtype,
                  vec_order=vec_order,
                  convert_to_directed=convert_to_directed)

  return Xout
