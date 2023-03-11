import numpy as np


def ground_truth_model(problem):
  X = np.array(problem["assignmentMatrix"], dtype=np.float32)
  return X


def dummy_model(problem):
  nP1 = problem["nP1"]
  nP2 = problem["nP2"]
  return np.eye(nP1, nP2)


def random_model(problem, rng=None):
  rng = np.random.default_rng(rng)
  nP1 = problem["nP1"]
  nP2 = problem["nP2"]
  out = np.eye(nP1, nP2)
  if out.shape[0] <= out.shape[1]:
    idx = rng.permutation(out.shape[0])
    out = out[idx, :]
  else:
    idx = rng.permutation(out.shape[1])
    out = out[:, idx]
  return out


def greedy_model(problem):
  def isok(pairs, res):
    for i, j in zip(*pairs):
      if res[i, j] < 0:
        return False
    # TODO: Whta if a hyperedge can have repeated nodes?
    for coords in pairs:
      if len(set(coords.flat)) != coords.size:
        return False
    return True

  nP1 = problem["nP1"]
  nP2 = problem["nP2"]
  valH3 = problem["valH3"]
  indH3 = problem["indH3"]
  res = np.zeros((nP1, nP2), dtype=np.int8)
  # Value in `valH3[sorted_inds]` is sort in descending order.
  sorted_inds = np.argsort(valH3, axis=None)[::-1]
  pairs_all = np.unravel_index(indH3[sorted_inds, :], res.shape)
  cnt = np.min(res.shape)
  for pairs in zip(*pairs_all):
    if cnt <= 0:
      break
    if isok(pairs, res):
      for i, j in zip(*pairs):
        if res[i, j] == 0:
          cnt -= 1
          res[i, :] = -1
          res[:, j] = -1
          res[i, j] = 1
  out = np.asarray(res > 0, dtype=np.float32)
  return out
