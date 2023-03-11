import numpy as np
from scipy.spatial import Delaunay
from scipy.spatial.distance import cdist

from .ann import annquery
from .compute_feature import compute_feature_simple


def _generate_triangles(nPoints, nTriangles, rng=None):
  """Randomly generate triangles"""
  rng = np.random.default_rng(rng)
  T = []
  for k in range(nTriangles):
    T.append(rng.choice(nPoints, size=3, replace=False))
  # T: nT x 3
  T = np.array(T, dtype=np.int32).reshape((-1, 3))
  return T


def _to_point_pair_index(indices, T1, T2, nP1, nP2):
  """Convert matching hyperedges to indices of matching point pairs

  Arguments:
    indices: nT1 x nNN, contains index of T2
    T1: nT1 x order, contains index of P1
    T2: nT2 x order, contains index of P2
    nP1: number of points in P1
    nP2: number of points in P2

  Return:
    ind: (nT1 * nNN) x order
  """
  assert indices.shape[0] == T1.shape[0]
  assert T1.shape[1] == T2.shape[1]
  indP1 = T1[..., np.newaxis, :]  # nT1 x 1 x order
  indP2 = T2[indices]             # nT1 x nNN x order
  # indP1 = np.broadcast_to(indP1, indP2.shape)  # nT1 x nNN x order; can be skipped
  ind = np.ravel_multi_index((indP1, indP2), (nP1, nP2))  # nT1 x nNN x order
  ind = np.reshape(ind, (-1, ind.shape[-1]))          # (nT1 * nNN) x order
  return ind


def _distances_to_affinity_scores(distances, scale=None):
  """
  distances: nT x nNN
  """
  if scale is None:
    scale = np.mean(distances)
  valH3 = np.exp(-distances.flat[:] / scale)
  return valH3


def _compute_affinity(feat1, feat2, T1, T2, nP1, nP2, nNN, scale=None):
  indices, distances = annquery(feat2, feat1, nNN)
  indH3 = _to_point_pair_index(indices, T1, T2, nP1, nP2)
  valH3 = _distances_to_affinity_scores(distances, scale)
  return indH3, valH3


def _sort_each_edge(indH, valH):
  assert indH.ndim == 2
  indH = np.sort(indH, axis=1)
  return indH, valH


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


def _select_edges_with_different_nodes(indH, valH, input_index_sorted=False):
  """
  Only select edges with different nodes, e.g. (i, j, k);
  edges like (i, i, i) and (i, i, j) will not be seletced.
  """
  assert indH.ndim == 2

  if input_index_sorted:
    indH_sorted = indH
  else:
    # Sort each edge
    indH_sorted = np.sort(indH, axis=1)
  selected_index = np.flatnonzero(np.all(indH_sorted[:, :-1] != indH_sorted[:, 1:], axis=1))
  # print(selected_index)  # TODO: for debug
  indH = indH[selected_index]
  valH = valH[selected_index]
  # print(indH)  # TODO: for debug
  # print(valH)  # TODO: for debug
  return indH, valH


def _sort_filter_unique(indH, valH):
  """
  1. Sort each edge;
  2. Only select edges with different nodes, e.g. (i, j, k);
     edges like (i, i, i) and (i, i, j) will be not be seletced.
  3. Unique all edges;

  Note: All edges will be sorted, too.
  """
  assert indH.ndim == 2

  # Sort each edge
  indH, valH = _sort_each_edge(indH, valH)
  # Filter out edges with same nodes
  indH, valH = _select_edges_with_different_nodes(indH, valH, input_index_sorted=True)
  # Unique all edges
  indH, valH = _unique_all_edges(indH, valH)
  return indH, valH


def create_feature_tensor(P1, P2, indH=None, T1=None, T2=None):
  """
  indH: N x order
  """
  nP1 = P1.shape[0]
  nP2 = P2.shape[0]
  if (T1 is None) or (T2 is None):
    T1, T2 = np.unravel_index(indH, shape=(nP1, nP2))
  feat1 = P1[T1].reshape((T1.shape[0], -1))
  feat2 = P2[T2].reshape((T2.shape[0], -1))
  featH = np.c_[feat1, feat2]
  return featH


def _build_complete_graph(points, dtype=np.float64):
  distance = cdist(points, points)
  A = distance.astype(dtype=dtype)

  indices = np.nonzero(A)
  distances = A[indices]

  indices = np.asarray(indices).T

  return indices, distances


def _build_delaunay_graph(points, dtype=np.float64):
  """
  Credit: Tao Wang, He Liu, Yidong Li, Yi Jin, Xiaohui Hou, and Haibin Ling. Learning combinatorial solver for graph matching. In CVPR, pages 7568–7577, 2020.
  """  # noqa

  npts = points.shape[0]
  A = np.zeros(shape=(npts, npts), dtype=dtype)
  distance = cdist(points, points)

  # Note:
  # In our experiments, some points may be collinear,
  # e.g. the last three points in `PascalVOC/annotations/horse/2009_002416_2.xml`.
  # To avoid the failure of Delaunay triangulation,
  # if #points <= 3, we will create a complete graph directly.
  # Though the number 3 works fine in our experiments,
  # more points (>3) may be collinear too.
  # So be careful if that happens!
  if points.shape[0] <= 3:
    A = distance.astype(dtype=dtype)
  else:
    # TODO: to deal with the extreme case that all points are collinear
    triangles = Delaunay(points).simplices
    for tri in triangles:
      A[tri[0]][tri[1]] = distance[tri[0]][tri[1]]
      A[tri[0]][tri[2]] = distance[tri[0]][tri[2]]
      A[tri[1]][tri[2]] = distance[tri[1]][tri[2]]
      A[tri[1]][tri[0]] = A[tri[0]][tri[1]]
      A[tri[2]][tri[0]] = A[tri[0]][tri[2]]
      A[tri[2]][tri[1]] = A[tri[1]][tri[2]]

  indices = np.nonzero(A)
  distances = A[indices]

  indices = np.asarray(indices).T

  return indices, distances


def generate_2nd_graph(points, dtype=np.float64):
  # return _build_complete_graph(points, dtype=dtype)
  return _build_delaunay_graph(points, dtype=dtype)


def create_tensor_2nd(P1, P2, scale):
  npts1 = len(P1)
  npts2 = len(P2)

  ind1, val1 = generate_2nd_graph(P1)
  ind2, val2 = generate_2nd_graph(P2)
  ne1 = len(ind1)
  ne2 = len(ind2)

  indices = np.tile(np.arange(ne2), reps=(ne1, 1))
  indH2 = _to_point_pair_index(indices, ind1, ind2, npts1, npts2)

  feat1 = np.reshape(val1, (ne1, 1, -1))
  feat2 = np.reshape(val2, (1, ne2, -1))
  dists = np.linalg.norm(feat1 - feat2, axis=-1)
  valH2 = _distances_to_affinity_scores(np.square(dists), scale)

  indH2, valH2 = _sort_filter_unique(indH2, valH2)

  return indH2, valH2


def generate_delaunay_triangles(points, dtype=np.int32):
  npts = len(points)
  if npts < 3:
    triangles = np.empty(shape=(0, 3), dtype=dtype)
  elif npts == 3:
    triangles = np.array([[0, 1, 2]], dtype=dtype)
  else:
    triangles = Delaunay(points).simplices
    triangles = triangles.astype(dtype=dtype)
  return triangles


def create_tensor_3rd_helper(P1, P2, T1, T2, nNN, scale=None, dtype=np.float64):
  nP1 = P1.shape[0]
  nP2 = P2.shape[0]

  # feat1: nT1 x 3
  feat1 = compute_feature_simple(P1, T1, dtype=dtype)
  # feat2: nT2 x 3
  feat2 = compute_feature_simple(P2, T2, dtype=dtype)

  indH3, valH3 = _compute_affinity(feat1, feat2, T1, T2, nP1, nP2, nNN, scale=scale)
  indH3 = indH3.astype(np.int32)
  valH3 = valH3.astype(dtype)

  indH3, valH3 = _sort_filter_unique(indH3, valH3)

  return indH3, valH3


def create_tensor_3rd_random_fc(P1, P2, scale=None, rng=None):
  """
  graph1: random triangle generation
  graph2: complete graph (fully connected) with ANN selection
  """
  nP1 = P1.shape[0]
  nP2 = P2.shape[0]

  # 3rd order tensor
  # nT = nP1 * 20  # # of triangles in graph 1
  nT = nP1 * nP2  # # of triangles in graph 1

  dtype = np.float64
  order = 3

  # T1: nT x 3
  T1 = _generate_triangles(nP1, nT, rng=rng)
  T1 = np.asarray(T1, dtype=np.int32)

  # T2: (nP2 * nP2 * nP2) x 3
  T2 = np.mgrid[(slice(nP2),) * order].reshape(order, -1).astype(np.int32).T

  # number of nearest neighbors used for each triangle (results can be bad if too low)
  # nNN = 500
  nNN = nT

  indH3, valH3 = create_tensor_3rd_helper(P1, P2, T1, T2, nNN, scale=scale, dtype=dtype)

  return indH3, valH3


def create_tensor_3rd_tri_tri(P1, P2, scale=None):
  """
  graph1: Delaunay triangulation
  graph2: Delaunay triangulation
  """
  dtype = np.float64
  T1 = generate_delaunay_triangles(P1, dtype=np.int32)
  T2 = generate_delaunay_triangles(P2, dtype=np.int32)
  nNN = len(T2)
  indH3, valH3 = create_tensor_3rd_helper(P1, P2, T1, T2, nNN, scale=scale, dtype=dtype)
  return indH3, valH3


def create_tensor_3rd_tri_fc(P1, P2, scale=None):
  """
  graph1: Delaunay triangulation
  graph2: complete graph (fully connected) with ANN selection
  """
  dtype = np.float64
  order = 3

  nP1 = P1.shape[0]
  nP2 = P2.shape[0]

  T1 = generate_delaunay_triangles(P1, dtype=np.int32)
  # T2: (nP2 * nP2 * nP2) x 3
  T2 = np.mgrid[(slice(nP2),) * order].reshape(order, -1).astype(np.int32).T

  # number of nearest neighbors used for each triangle (results can be bad if too low)
  nNN = nP1 * nP2

  indH3, valH3 = create_tensor_3rd_helper(P1, P2, T1, T2, nNN, scale=scale, dtype=dtype)

  return indH3, valH3


def create_tensor_3rd_tri_random_fc(P1, P2, scale=None, rng=None):
  """
  graph1: union of Delaunay triangulation and random triangle generation
  graph2: complete graph (fully connected) with ANN selection
  """
  nP1 = P1.shape[0]
  nP2 = P2.shape[0]

  dtype = np.float64
  order = 3

  # T1: nT x 3
  T1_rand = _generate_triangles(nP1, nTriangles=nP1 * nP2, rng=rng)
  T1_tri = generate_delaunay_triangles(P1, dtype=np.int32)
  T1 = np.concatenate((T1_rand, T1_tri), axis=0)

  T1 = np.asarray(T1, dtype=np.int32)

  # T2: (nP2 * nP2 * nP2) x 3
  T2 = np.mgrid[(slice(nP2),) * order].reshape(order, -1).astype(np.int32).T

  # number of nearest neighbors used for each triangle (results can be bad if too low)
  nNN = nP1 * nP2

  indH3, valH3 = create_tensor_3rd_helper(P1, P2, T1, T2, nNN, scale=scale, dtype=dtype)

  return indH3, valH3


def create_tensor_3rd(P1, P2, scale=None, rng=None):
  return create_tensor_3rd_random_fc(P1, P2, scale=scale, rng=rng)
  # return create_tensor_3rd_tri_tri(P1, P2, scale=scale)
  # return create_tensor_3rd_tri_fc(P1, P2, scale=scale)
  # return create_tensor_3rd_tri_random_fc(P1, P2, scale=scale, rng=rng)
