import collections.abc
from copy import deepcopy
import itertools
import numpy as np

from .create_tensor import create_tensor_2nd
from .create_tensor import create_tensor_3rd
from .create_tensor import _unique_all_edges


def create_problem(P1, P2, assignmentMatrix, scale=None,
                   image1=None, image2=None,
                   cid=None,
                   order=3,
                   rng=None):
  """
  Input:
    order: 2, 3, (2, 3), or [2, 3]

  Output:
    problem: dict
    {
      cid:    int or None
      nP1:    int
      nP2:    int
      P1:     float64, (nP1, 2)
      P2:     float64, (nP2, 2)
      indH1:  int32,   (N1, 1) array or None
      valH1:  float64, (N1,)   array or None
      indH2:  int32,   (N2, 2) array or None
      valH2:  float64, (N2,)   array or None
      indH3:  int32,   (N3, 3) array or None
      valH3:  float64, (N3,)   array or None
      image1: uint8, (height, width, channel) array or None
      image2: uint8, (height, width, channel) array or None
      assignmentMatrix: bool, (nP1, nP2)
    }
  """
  if not isinstance(order, collections.Sequence):
    order = (order,)

  nP1 = P1.shape[0]
  nP2 = P2.shape[0]

  # 1st order tensor
  indH1 = None  # TODO: Or np.zeros((0, 1), dtype=np.int32)?
  valH1 = None  # TODO: Or np.zeros((0, 1), dtype=np.float64)?

  # 2nd order tensor
  indH2 = None  # TODO: Or np.zeros((0, 2), dtype=np.int32)?
  valH2 = None  # TODO: Or np.zeros((0, 1), dtype=np.float64)?

  indH3 = None  # TODO: Or np.zeros((0, 3), dtype=np.int32)?
  valH3 = None  # TODO: Or np.zeros((0, 1), dtype=np.float64)?

  if 2 in order:
    # TODO: use different scales for different order
    indH2, valH2 = create_tensor_2nd(P1, P2, scale=scale)

  if 3 in order:
    indH3, valH3 = create_tensor_3rd(P1, P2, scale=scale, rng=rng)

  # Note(Xiaowei):
  #   `indH3` start from 0 (same as the original matlab code)
  problem = {
      "cid": cid,
      "nP1": nP1,
      "nP2": nP2,
      "P1": P1,
      "P2": P2,
      "indH1": indH1,
      "valH1": valH1,
      "indH2": indH2,
      "valH2": valH2,
      "indH3": indH3,
      "valH3": valH3,
      "image1": image1,
      "image2": image2,
      "assignmentMatrix": assignmentMatrix
  }

  return problem


def _undirected_to_directed(indH, valH, unique_and_sort=True):
  order = indH.shape[-1]
  indices = np.asarray(list(itertools.permutations(range(order))))
  indH = indH[:, indices].reshape(-1, order)
  N = len(indices)
  valH = np.repeat(valH, repeats=N, axis=0)
  if unique_and_sort:
    indH, valH = _unique_all_edges(indH, valH)
  return indH, valH


def convert_to_directed(problem, unique_and_sort=True, copy=True):
  """
  Note: It's supposed that each input edge are undirected.
  """
  if copy:
    problem = deepcopy(problem)

  def update(ind_key, val_key):
    indH = problem[ind_key]
    valH = problem[val_key]
    if indH is not None:
      indH, valH = _undirected_to_directed(indH, valH, unique_and_sort=unique_and_sort)
      problem[ind_key] = indH
      problem[val_key] = valH

  # Note: `indH1` and `valH1` remain the same
  update("indH2", "valH2")
  update("indH3", "valH3")

  return problem
