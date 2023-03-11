import numpy as np

from .bcagm_basic import bcagm as bcagm_lin
from .bcagm_basic import bcagm_quad
from .bcagm_basic import bcagm3 as bcagm3_lin
from .bcagm_basic import bcagm3_quad


def bcagm(problem, max_iter=50):
  """
  Output:
    X: nP1 x nP2 (Note: this is different from `bcagm_basic.py`)
  """
  nP1 = problem["nP1"]
  nP2 = problem["nP2"]
  X0 = np.ones([4 * nP2 * nP1, 1], dtype=np.float32)
  Xout = bcagm_lin(problem, X0, max_iter=max_iter, adapt=0)

  # N2 x N1 => N1 x N2
  # TODO: directly output (N1, N2) matrix
  Xout = Xout.T

  return Xout


def adapt_bcagm(problem, max_iter=50):
  """
  Output:
    X: nP1 x nP2 (Note: this is different from `bcagm_basic.py`)
  """
  nP1 = problem["nP1"]
  nP2 = problem["nP2"]
  X0 = np.ones([4 * nP2 * nP1, 1], dtype=np.float32)
  Xout = bcagm_lin(problem, X0, max_iter=max_iter, adapt=1)

  # N2 x N1 => N1 x N2
  # TODO: directly output (N1, N2) matrix
  Xout = Xout.T

  return Xout


def bcagm_mp(problem):
  """
  Output:
    X: nP1 x nP2 (Note: this is different from `bcagm_basic.py`)
  """
  nP1 = problem["nP1"]
  nP2 = problem["nP2"]
  X0 = np.ones([2 * nP2 * nP1, 1], dtype=np.float32)
  Xout = bcagm_quad(problem, X0, subroutine=1, adapt=0)

  # N2 x N1 => N1 x N2
  # TODO: directly output (N1, N2) matrix
  Xout = Xout.T

  return Xout


def adapt_bcagm_mp(problem):
  """
  Output:
    X: nP1 x nP2 (Note: this is different from `bcagm_basic.py`)
  """
  nP1 = problem["nP1"]
  nP2 = problem["nP2"]
  X0 = np.ones([2 * nP2 * nP1, 1], dtype=np.float32)
  Xout = bcagm_quad(problem, X0, subroutine=1, adapt=1)

  # N2 x N1 => N1 x N2
  # TODO: directly output (N1, N2) matrix
  Xout = Xout.T

  return Xout


def bcagm_ipfp(problem):
  """
  Output:
    X: nP1 x nP2 (Note: this is different from `bcagm_basic.py`)
  """
  nP1 = problem["nP1"]
  nP2 = problem["nP2"]
  X0 = np.ones([2 * nP2 * nP1, 1], dtype=np.float32)
  Xout = bcagm_quad(problem, X0, subroutine=2, adapt=0)

  # N2 x N1 => N1 x N2
  # TODO: directly output (N1, N2) matrix
  Xout = Xout.T

  return Xout


def adapt_bcagm_ipfp(problem):
  """
  Output:
    X: nP1 x nP2 (Note: this is different from `bcagm_basic.py`)
  """
  nP1 = problem["nP1"]
  nP2 = problem["nP2"]
  X0 = np.ones([2 * nP2 * nP1, 1], dtype=np.float32)
  Xout = bcagm_quad(problem, X0, subroutine=2, adapt=1)

  # N2 x N1 => N1 x N2
  # TODO: directly output (N1, N2) matrix
  Xout = Xout.T

  return Xout


def bcagm3(problem, max_iter=50):
  """
  Output:
    X: nP1 x nP2 (Note: this is different from `bcagm_basic.py`)
  """
  nP1 = problem["nP1"]
  nP2 = problem["nP2"]
  X0 = np.ones([3 * nP2 * nP1, 1], dtype=np.float32)
  Xout = bcagm3_lin(problem, X0, max_iter=max_iter, adapt=0)

  # N2 x N1 => N1 x N2
  # TODO: directly output (N1, N2) matrix
  Xout = Xout.T

  return Xout


def adapt_bcagm3(problem, max_iter=50):
  """
  Output:
    X: nP1 x nP2 (Note: this is different from `bcagm_basic.py`)
  """
  nP1 = problem["nP1"]
  nP2 = problem["nP2"]
  X0 = np.ones([3 * nP2 * nP1, 1], dtype=np.float32)
  Xout = bcagm3_lin(problem, X0, max_iter=max_iter, adapt=1)

  # N2 x N1 => N1 x N2
  # TODO: directly output (N1, N2) matrix
  Xout = Xout.T

  return Xout


def bcagm3_mp(problem):
  """
  Output:
    X: nP1 x nP2 (Note: this is different from `bcagm_basic.py`)
  """
  nP1 = problem["nP1"]
  nP2 = problem["nP2"]
  X0 = np.ones([2 * nP2 * nP1, 1], dtype=np.float32)
  Xout = bcagm3_quad(problem, X0, subroutine=1, adapt=0)

  # N2 x N1 => N1 x N2
  # TODO: directly output (N1, N2) matrix
  Xout = Xout.T

  return Xout


def adapt_bcagm3_mp(problem):
  """
  Output:
    X: nP1 x nP2 (Note: this is different from `bcagm_basic.py`)
  """
  nP1 = problem["nP1"]
  nP2 = problem["nP2"]
  X0 = np.ones([2 * nP2 * nP1, 1], dtype=np.float32)
  Xout = bcagm3_quad(problem, X0, subroutine=1, adapt=1)

  # N2 x N1 => N1 x N2
  # TODO: directly output (N1, N2) matrix
  Xout = Xout.T

  return Xout


def bcagm3_ipfp(problem):
  """
  Output:
    X: nP1 x nP2 (Note: this is different from `bcagm_basic.py`)
  """
  nP1 = problem["nP1"]
  nP2 = problem["nP2"]
  X0 = np.ones([2 * nP2 * nP1, 1], dtype=np.float32)
  Xout = bcagm3_quad(problem, X0, subroutine=2, adapt=0)

  # N2 x N1 => N1 x N2
  # TODO: directly output (N1, N2) matrix
  Xout = Xout.T

  return Xout


def adapt_bcagm3_ipfp(problem):
  """
  Output:
    X: nP1 x nP2 (Note: this is different from `bcagm_basic.py`)
  """
  nP1 = problem["nP1"]
  nP2 = problem["nP2"]
  X0 = np.ones([2 * nP2 * nP1, 1], dtype=np.float32)
  Xout = bcagm3_quad(problem, X0, subroutine=2, adapt=1)

  # N2 x N1 => N1 x N2
  # TODO: directly output (N1, N2) matrix
  Xout = Xout.T

  return Xout
