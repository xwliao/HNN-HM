import itertools
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.sparse import coo_matrix


def _debug_print(s, *args, **kwargs):
    # print(s, *args, **kwargs)
    pass


# def mat2vec(m, dtype=None):
#   v = np.array(m.flat[:], dtype=dtype)
#   return v


# def vec2mat(v, n1, n2, dtype=None):
#   m = np.array(v.reshape((n1, n2)), dtype=dtype)
#   return m


def mat2vec(m, dtype=None, order='C'):
  # assert order in ['C', 'F']
  v = np.reshape(m, m.size, order=order)
  v = np.asarray(v, dtype=dtype)
  return v


def vec2mat(v, n1, n2, dtype=None, order='C'):
  # assert order in ['C', 'F']
  m = np.reshape(v, (n1, n2), order=order)
  m = np.asarray(m, dtype=dtype)
  return m


def quadratic_form(A, x):
  """
  x' * A * x
  """
  # q = np.dot(np.dot(x, A), x)
  q = np.linalg.multi_dot([x, A, x])
  return q


def hungarian(affinity: np.ndarray, dtype=None):
  """
  Solve optimal LAP permutation by hungarian algorithm.
  :param affinity: input 2d tensor
  :return: optimal permutation matrix
  """
  row, col = linear_sum_assignment(affinity, maximize=True)

  perm_mat = np.zeros_like(affinity, dtype=dtype)
  perm_mat[row, col] = 1

  return perm_mat


def project(xv, n1, n2, order='C'):
  dtype = xv.dtype
  xm = vec2mat(xv, n1, n2, order=order)
  bm = hungarian(xm, dtype=dtype)
  bv = mat2vec(bm, order=order)
  return bv


def line_search(A, b, x, dx):
  """
  argmax_t 0.5 * x(t)' * A * x(t) + b' * x(t)
  s.t.
    x(t) = x + t * dx
    t in [0, 1]

  require:
  1. A is symmetric
  2. (A * x + b)' * dx >= 0
  """
  # D = np.dot(np.dot(dx, A), dx)       # dx' * A * dx
  D = quadratic_form(A, dx)       # dx' * A * dx
  if D < 0:
    C = np.dot(np.dot(A, x) + b, dx)   # (A * x + b)' * dx
    t = min(C / -D, 1.0)
  else:
    t = 1.0
  return t


def _ipfp_2nd_helper(M, d, x0, n1, n2, score_call_back, max_iter,
                     x_opt=None, s_opt=None,
                     eps=1e-6, step_eps=0., order='C'):
  """
  x'Mx + d'x

  if step <= step_eps:
    step = 0

  require M is symmetric
  """
  def update_opt(score_opt, x_opt, x_new):
    # _debug_print(f'x_new: {x_new}')             # TODO: just for debug
    score_new = score_call_back(x_new)
    _debug_print(f'new score: {score_new}')   # TODO: just for debug
    if score_new > score_opt:
      score_opt = score_new
      x_opt = x_new
      _debug_print(f'new opt score: {score_opt}')   # TODO: just for debug
      # _debug_print(f'new opt x: {x_opt}')           # TODO: just for debug
    return score_opt, x_opt

  # Change to 0.5 * x' * M * x + 0.5 * d' * x
  d2 = 0.5 * d

  if x_opt is None:
      x_opt = x0
      s_opt = score_call_back(x_opt)
  x = x0
  for _ in range(max_iter):
    _debug_print(f'x: {x[:12]}')             # TODO: just for debug
    g = np.dot(M, x) + d2
    _debug_print(f'g: {g[:12]}')             # TODO: just for debug
    b = project(g, n1, n2, order=order)
    _debug_print(f'b: {b[:12]}')             # TODO: just for debug
    s_opt, x_opt = update_opt(s_opt, x_opt, b)
    dx = b - x
    t = line_search(M, d2, x, dx)
    if t <= step_eps:
      t = 0.
    _debug_print(f't: {t}')                     # TODO: just for debug
    tdx = t * dx
    x = x + tdx
    if np.linalg.norm(tdx) < eps:
      break

  return x, x_opt, s_opt


def ipfp_2nd(M, d, X0, max_iter=50,
             eps=1e-12, step_eps=0.01,
             internal_dtype='float64',
             vec_order='C'):
  """
  x'Mx + d'x

  require M is symmetric
  """
  def score_call_back(x):
    return np.dot((np.dot(M, x) + d), x)

  assert len(X0.shape) == 2
  n1, n2 = X0.shape
  out_dtype = X0.dtype

  # # XXX:
  # X0 = X0 / np.linalg.norm(X0)

  M = np.asarray(M, dtype=internal_dtype)
  d = np.asarray(d, dtype=internal_dtype)
  x = mat2vec(X0, dtype=internal_dtype, order=vec_order)
  _, x_opt, _ = _ipfp_2nd_helper(
      M, d, x, n1, n2,
      score_call_back=score_call_back,
      max_iter=max_iter,
      eps=eps, step_eps=step_eps,
      order=vec_order
  )

  X = vec2mat(x_opt, n1, n2, dtype=out_dtype, order=vec_order)
  return X


def tensor_dot(indH, valH, x, dtype=None):
    """
    H: 3rd order potential (may not be super-symmetric)
    x: assignment vector
    """
    ind1 = indH[:, 0]
    ind2 = indH[:, 1]
    ind3 = indH[:, 2]
    v = valH.flat * x[ind3]
    n = x.size
    coo = coo_matrix((v, (ind1, ind2)), shape=(n, n), dtype=dtype)
    m = coo.toarray()
    return m


def _unique_all_edges(indH, valH):
  """
  Note: Order in each row of `indH` matters.
        So two rows with some set of values but in a different order
        will be consider different.

  Note: The returned indH will be sorted.

  Note: If same indices have different values,
        current implementation will choose only one of them.
  """
  if indH.size == 0:
    return indH, valH
  indH, indices = np.unique(indH, return_index=True, axis=0)
  # TODO: What if same indices have different values?
  valH = valH[indices]
  return indH, valH


def _undirected_to_directed(indH, valH, unique_and_sort=True):
  order = indH.shape[-1]
  indices = np.asarray(list(itertools.permutations(range(order))))
  indH = indH[:, indices].reshape(-1, order)
  N = len(indices)
  valH = np.repeat(valH, repeats=N, axis=0)
  if unique_and_sort:
    indH, valH = _unique_all_edges(indH, valH)
  return indH, valH


def ipfp_3rd(indH, valH, X0,
             max_iter,
             max_iter_2nd,
             eps=1e-12,
             step_eps=0.,
             internal_dtype='float64',
             vec_order='C',
             convert_to_directed=True):

  if convert_to_directed:
    indH, valH = _undirected_to_directed(indH, valH, unique_and_sort=True)

  def score_call_back(x):
    """
    H: 3rd order potential (super-symmetric)
    x: assignment vector
    """
    T = x[indH]
    return np.einsum('i,i,i,i->', valH.flat, T[:, 0], T[:, 1], T[:, 2])

  assert len(X0.shape) == 2
  n1, n2 = X0.shape
  out_dtype = X0.dtype

  # # XXX:
  # X0 = X0 / np.linalg.norm(X0)

  x = mat2vec(X0, dtype=internal_dtype, order=vec_order)

  x_opt = x
  s_opt = score_call_back(x_opt)

  for _ in range(max_iter):
    _debug_print(f'x_opt: {x_opt}')  # TODO: just for debug
    _debug_print(f's_opt: {s_opt}')  # TODO: just for debug

    # XXX:
    M = tensor_dot(indH, valH, x, dtype=internal_dtype)
    M = 0.5 * (M + M.T)  # convert to symmetric matrix

    d = np.dot(M, x)

    # NOTE: don't forget the minus sign before `d`
    x, x_opt_new, s_opt_new = _ipfp_2nd_helper(
        M, -d, x, n1, n2,
        score_call_back=score_call_back,
        max_iter=max_iter_2nd,
        x_opt=x_opt,
        s_opt=s_opt,
        eps=eps,
        step_eps=step_eps,
        order=vec_order
    )

    _debug_print(f'after, x_opt_new: {x_opt_new}')  # TODO: just for debug
    _debug_print(f'after, s_opt_new: {s_opt_new}')  # TODO: just for debug

    if s_opt_new <= s_opt:
      break
    x_opt = x_opt_new
    s_opt = s_opt_new
    _debug_print(f'after, x_opt: {x_opt}')  # TODO: just for debug
    _debug_print(f'after, s_opt: {s_opt}')  # TODO: just for debug

  X = vec2mat(x_opt, n1, n2, dtype=out_dtype, order=vec_order)
  return X
