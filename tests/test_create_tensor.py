import numpy as np
from utils.create_tensor import _sort_each_edge
from utils.create_tensor import _unique_all_edges
from utils.create_tensor import _select_edges_with_different_nodes
from utils.create_tensor import _sort_filter_unique
from utils.create_tensor import _to_point_pair_index
from utils.create_tensor import _distances_to_affinity_scores
from utils.create_tensor import create_tensor_3rd_helper
from utils.create_tensor import create_feature_tensor
from utils.create_tensor import create_tensor_2nd
from utils.create_tensor import create_tensor_3rd
from utils.create_tensor import create_tensor_3rd_random_fc
from utils.create_tensor import create_tensor_3rd_tri_tri
from utils.create_tensor import create_tensor_3rd_tri_fc
from utils.create_tensor import create_tensor_3rd_tri_random_fc


def read_data(name, dtype=None):
  return np.loadtxt('test_data/{}.txt'.format(name), dtype=dtype)


def get_saved_data(dtype=np.float64):
  P1 = read_data('P1', dtype=dtype).T  # nP1 x 2
  P2 = read_data('P2', dtype=dtype).T  # nP2 x 2
  nP1 = P1.shape[0]
  nP2 = P2.shape[0]

  indH3 = read_data('indH3', dtype=np.int32)
  valH3 = read_data('valH3', dtype=np.float64)

  T1 = read_data('T1', dtype=np.int32).T  # nT x 3
  T2 = read_data('T2', dtype=np.int32).T  # nP2*nP2*nP2 x 3

  nT1 = T1.shape[0]
  nNN = nT1

  indices = read_data('inds', dtype=np.int32).T - 1  # nT x nNN
  distances = read_data('dists').T  # nT x nNN

  data = {
      "nP1": nP1,
      "nP2": nP2,
      "P1": P1,
      "P2": P2,
      "indH3": indH3,
      "valH3": valH3,
      "T1": T1,
      "T2": T2,
      "nNN": nNN,
      "scale": 0.2,
      "indices": indices,
      "distances": distances
  }
  return data


def test_distances_to_affinity_scores():
  distances = np.array([1., 3.], dtype=np.float32)
  val = _distances_to_affinity_scores(distances, scale=None)
  val_t = np.exp([-0.5, -1.5])
  np.testing.assert_allclose(val, val_t)

  distances = np.array([1, 3], dtype=np.int64)
  val = _distances_to_affinity_scores(distances, scale=None)
  val_t = np.exp([-0.5, -1.5])
  np.testing.assert_allclose(val, val_t)

  distances = np.array([1., 3.], dtype=np.float32)
  val = _distances_to_affinity_scores(distances, scale=0.2)
  val_t = np.exp([-5., -15.])
  np.testing.assert_allclose(val, val_t)


def test_sort_each_edge_1st():
  ind = np.array([[4],
                  [1],
                  [0],
                  [1]],
                 dtype=np.int32)

  val = np.array([[0.6],
                  [0.9],
                  [0.4],
                  [0.9]],
                 dtype=np.float64)

  ind2, val2 = _sort_each_edge(ind, val)

  np.testing.assert_allclose(ind2, ind)
  np.testing.assert_allclose(val2, val)


def test_sort_each_edge_2nd():
  ind = np.array([[4, 0],
                  [1, 0],
                  [0, 1],
                  [1, 0]],
                 dtype=np.int32)

  val = np.array([[0.6],
                  [0.9],
                  [0.4],
                  [0.4]],
                 dtype=np.float64)

  ind2, val2 = _sort_each_edge(ind, val)

  ind_t = np.array([[0, 4],
                    [0, 1],
                    [0, 1],
                    [0, 1]],
                   dtype=np.int32)

  np.testing.assert_allclose(ind2, ind_t)
  np.testing.assert_allclose(val2, val)


def test_sort_each_edge_3rd():
  ind = np.array([[4, 0, 2],
                  [1, 0, 6],
                  [4, 0, 2],
                  [0, 1, 3]],
                 dtype=np.int32)

  val = np.array([[0.6],
                  [0.9],
                  [0.6],
                  [0.4]],
                 dtype=np.float64)

  ind2, val2 = _sort_each_edge(ind, val)

  ind_t = np.array([[0, 2, 4],
                    [0, 1, 6],
                    [0, 2, 4],
                    [0, 1, 3]],
                   dtype=np.int32)

  np.testing.assert_allclose(ind2, ind_t)
  np.testing.assert_allclose(val2, val)


def test_unique_all_edges_1st():
  ind = np.array([[4],
                  [6],
                  [4],
                  [3]],
                 dtype=np.int32)

  val = np.array([[0.6],
                  [0.9],
                  [0.6],
                  [0.4]],
                 dtype=np.float64)

  ind2, val2 = _unique_all_edges(ind, val)

  ind_t = np.array([[3],
                    [4],
                    [6]],
                   dtype=np.int32)

  val_t = np.array([[0.4],
                    [0.6],
                    [0.9]],
                   dtype=np.float64)

  np.testing.assert_allclose(ind2, ind_t)
  np.testing.assert_allclose(val2, val_t)


def test_unique_all_edges_2nd():
  ind = np.array([[0, 2],
                  [0, 1],
                  [0, 2],
                  [0, 1]],
                 dtype=np.int32)

  val = np.array([[0.6],
                  [0.9],
                  [0.6],
                  [0.9]],
                 dtype=np.float64)

  ind2, val2 = _unique_all_edges(ind, val)

  ind_t = np.array([[0, 1],
                    [0, 2]],
                   dtype=np.int32)

  val_t = np.array([[0.9],
                    [0.6]],
                   dtype=np.float64)

  np.testing.assert_allclose(ind2, ind_t)
  np.testing.assert_allclose(val2, val_t)


def test_unique_all_edges_3rd():
  ind = np.array([[0, 2, 4],
                  [0, 1, 6],
                  [0, 2, 4],
                  [0, 1, 3]],
                 dtype=np.int32)

  val = np.array([[0.6],
                  [0.9],
                  [0.6],
                  [0.4]],
                 dtype=np.float64)

  ind2, val2 = _unique_all_edges(ind, val)

  ind_t = np.array([[0, 1, 3],
                    [0, 1, 6],
                    [0, 2, 4]],
                   dtype=np.int32)

  val_t = np.array([[0.4],
                    [0.9],
                    [0.6]],
                   dtype=np.float64)

  np.testing.assert_allclose(ind2, ind_t)
  np.testing.assert_allclose(val2, val_t)


def test_unique_all_edges_3rd_directed():
  ind = np.array([[0, 1, 2],  # 0
                  [0, 2, 1],  # 1
                  [2, 0, 1],  # 2
                  [1, 2, 0],  # 3
                  [1, 0, 2],  # 4
                  [2, 1, 0]],  # 5
                 dtype=np.int32)

  val = np.arange(len(ind), dtype=np.float64).reshape((-1, 1))

  ind_t = np.array([[0, 1, 2],
                    [0, 2, 1],
                    [1, 0, 2],
                    [1, 2, 0],
                    [2, 0, 1],
                    [2, 1, 0]],
                   dtype=np.int32)

  val_t = np.array([[0],
                    [1],
                    [4],
                    [3],
                    [2],
                    [5]],
                   dtype=np.float64)

  ind2, val2 = _unique_all_edges(ind, val)
  np.testing.assert_allclose(ind2, ind_t)
  np.testing.assert_allclose(val2, val_t)


def test_select_edges_with_different_nodes():
  ind = np.array([[0, 1, 2],  # 0; OK
                  [0, 0, 0],  # 1; (i, i, i)
                  [1, 1, 1],  # 2; (i, i, i)
                  [0, 2, 1],  # 3; OK
                  [0, 0, 1],  # 4; (i, i, j)
                  [0, 1, 0],  # 5; (i, j, i)
                  [1, 0, 0],  # 6; (j, i, i)
                  [0, 1, 1],  # 7; (i, j, j)
                  [1, 0, 1],  # 8; (j, i, j)
                  [1, 1, 0],  # 9; (j, j, i)
                  [1, 2, 0],  # 10; OK
                  [2, 1, 0]],  # 11; OK
                 dtype=np.int32)

  val = np.arange(len(ind), dtype=np.float64).reshape((-1, 1))

  ind_t = np.array([[0, 1, 2],
                    [0, 2, 1],
                    [1, 2, 0],
                    [2, 1, 0]],
                   dtype=np.int32)

  val_t = np.array([[0],
                    [3],
                    [10],
                    [11]],
                   dtype=np.float64)

  ind2, val2 = _select_edges_with_different_nodes(ind, val, input_index_sorted=False)
  np.testing.assert_allclose(ind2, ind_t)
  np.testing.assert_allclose(val2, val_t)


def test_select_edges_with_different_nodes_2():
  ind = np.array([[0, 0, 0],
                  [1, 1, 1],
                  [0, 0, 1],
                  [0, 1, 0],
                  [1, 0, 0],
                  [0, 1, 1],
                  [1, 0, 1],
                  [1, 1, 0]],
                 dtype=np.int32)

  val = np.arange(len(ind), dtype=np.float64).reshape((-1, 1))

  ind_t = np.empty(shape=(0, 3), dtype=ind.dtype)
  val_t = np.empty(shape=(0, 1), dtype=val.dtype)

  ind2, val2 = _select_edges_with_different_nodes(ind, val, input_index_sorted=False)

  np.testing.assert_allclose(ind2, ind_t)
  np.testing.assert_allclose(val2, val_t)


def test_select_edges_with_different_nodes_sorted():
  ind = np.array([[0, 1, 2],  # 0; OK
                  [0, 0, 0],  # 1; (i, i, i)
                  [1, 1, 1],  # 2; (i, i, i)
                  [0, 0, 1],  # 3; (i, i, j)
                  [0, 1, 1]],  # 4; (i, j, j)
                 dtype=np.int32)

  val = np.arange(len(ind), dtype=np.float64).reshape((-1, 1))

  ind_t = np.array([[0, 1, 2]], dtype=np.int32)

  val_t = np.array([[0]], dtype=np.float64)

  ind2, val2 = _select_edges_with_different_nodes(ind, val, input_index_sorted=True)
  np.testing.assert_allclose(ind2, ind_t)
  np.testing.assert_allclose(val2, val_t)


def test_select_edges_with_different_nodes_sorted_2():
  ind = np.array([[0, 0, 0],
                  [1, 1, 1],
                  [0, 0, 1],
                  [0, 1, 1]],
                 dtype=np.int32)

  val = np.arange(len(ind), dtype=np.float64).reshape((-1, 1))

  ind_t = np.empty(shape=(0, 3), dtype=ind.dtype)
  val_t = np.empty(shape=(0, 1), dtype=val.dtype)

  ind2, val2 = _select_edges_with_different_nodes(ind, val, input_index_sorted=True)
  np.testing.assert_allclose(ind2, ind_t)
  np.testing.assert_allclose(val2, val_t)


def test_sort_filter_unique():
  ind = np.array([[0, 1, 2],  # 0; OK
                  [0, 0, 0],  # 1; (i, i, i)
                  [1, 1, 1],  # 2; (i, i, i)
                  [0, 2, 1],  # 3; unsorted
                  [0, 0, 1],  # 4; (i, i, j)
                  [0, 1, 0],  # 5; (i, j, i)
                  [1, 0, 0],  # 6; (j, i, i)
                  [0, 1, 1],  # 7; (i, j, j)
                  [1, 0, 1],  # 8; (j, i, j)
                  [1, 1, 0],  # 9; (j, j, i)
                  [1, 2, 0],  # 10; unsorted
                  [2, 1, 0],  # 11; unsorted
                  [1, 2, 3]],  # 12; OK
                 dtype=np.int32)

  val = np.arange(len(ind), dtype=np.float64).reshape((-1, 1))
  val[0] = 100
  val[3] = 100
  val[10] = 100
  val[11] = 100

  ind_t = np.array([[0, 1, 2],
                    [1, 2, 3]],
                   dtype=np.int32)

  val_t = np.array([[100],
                    [12]],
                   dtype=np.float64)

  ind2, val2 = _sort_filter_unique(ind, val)
  np.testing.assert_allclose(ind2, ind_t)
  np.testing.assert_allclose(val2, val_t)


def test_sort_filter_unique_2():
  ind = np.array([[0, 0, 0],
                  [1, 1, 1],
                  [0, 0, 1],
                  [0, 1, 0],
                  [1, 0, 0],
                  [0, 1, 1],
                  [1, 0, 1],
                  [1, 1, 0]],
                 dtype=np.int32)

  val = np.arange(len(ind), dtype=np.float64).reshape((-1, 1))

  ind_t = np.empty(shape=(0, 3), dtype=ind.dtype)
  val_t = np.empty(shape=(0, 1), dtype=val.dtype)

  ind2, val2 = _sort_filter_unique(ind, val)
  np.testing.assert_allclose(ind2, ind_t)
  np.testing.assert_allclose(val2, val_t)


def test_indH3():
  data = get_saved_data()
  nP1 = data["nP1"]
  nP2 = data["nP2"]
  T1 = data["T1"]
  T2 = data["T2"]
  indices = data["indices"]
  indH3_target = data["indH3"]

  indH3 = _to_point_pair_index(indices, T1, T2, nP1, nP2)

  np.testing.assert_allclose(indH3, indH3_target)


def test_valH3():
  data = get_saved_data()
  scale = data["scale"]
  distances = data["distances"]
  valH3_target = data["valH3"]

  valH3 = _distances_to_affinity_scores(distances, scale)

  np.testing.assert_allclose(valH3, valH3_target)


def test_create_feature_tensor():
  P1 = np.array([[0, 0],
                 [0, 2],
                 [3, 0]], dtype=np.float32)
  P2 = np.array([[-1, 0],
                 [0, 3],
                 [2, 0],
                 [1, -1]], dtype=np.float32)

  T1 = np.array([[0, 1, 2],
                 [2, 0, 1]], dtype=np.int32)
  T2 = np.array([[0, 1, 2],
                 [3, 2, 0]], dtype=np.int32)

  indH = np.array([[0, 5, 10],
                   [11, 2, 4]], dtype=np.int32)

  featH_target = np.array([[0, 0, 0, 2, 3, 0, -1, 0, 0, 3, 2, 0],
                           [3, 0, 0, 0, 0, 2, 1, -1, 2, 0, -1, 0]], dtype=np.float32)

  featH = create_feature_tensor(P1, P2, indH=indH)
  np.testing.assert_allclose(featH, featH_target)

  featH = create_feature_tensor(P1, P2, T1=T1, T2=T2)
  np.testing.assert_allclose(featH, featH_target)


def test_create_feature_tensor_indH():
  d = 2
  n1 = 10
  n2 = 20
  P1 = np.random.rand(n1, d)
  P2 = np.random.rand(n2, d)

  N = 20
  order = 3
  indH = np.random.choice(n1 * n2, size=(N, order))

  featH = create_feature_tensor(P1, P2, indH=indH)
  assert featH.shape == (N, 2 * order * d)


def test_create_feature_tensor_T1_T2():
  d = 2
  n1 = 10
  n2 = 20
  P1 = np.random.rand(n1, d)
  P2 = np.random.rand(n2, d)

  N = 20
  order = 3
  T1 = np.random.choice(n1, size=(N, order))
  T2 = np.random.choice(n2, size=(N, order))

  featH = create_feature_tensor(P1, P2, T1=T1, T2=T2)
  assert featH.shape == (N, 2 * order * d)


def test_create_tensor_2nd():
  P1 = np.random.rand(10, 2)
  P2 = np.random.rand(20, 2)
  indH2, valH2 = create_tensor_2nd(P1, P2, scale=1.)
  assert indH2.shape[0] == valH2.shape[0]


def test_create_tensor_3rd_helper():
  """Run `create_tensor_3rd_helper` and compare its result with the saved result generated by the Matlab code"""

  data = get_saved_data()
  P1 = data["P1"]
  P2 = data["P2"]
  T1 = data["T1"]
  T2 = data["T2"]
  nNN = data["nNN"]
  scale = data["scale"]
  dtype = data["valH3"].dtype
  indH3_target = data["indH3"]
  valH3_target = data["valH3"]

  # TODO: test sort_each_edge and unique_all_edges
  indH3_target, valH3_target = _sort_each_edge(indH3_target, valH3_target)
  indH3_target, valH3_target = _unique_all_edges(indH3_target, valH3_target)

  indH3, valH3 = create_tensor_3rd_helper(P1, P2, T1, T2, nNN, scale, dtype=dtype)

  np.testing.assert_allclose(indH3, indH3_target, atol=1e-6)
  np.testing.assert_allclose(valH3, valH3_target, atol=1e-6)


def test_create_tensor_3rd():
  P1 = np.random.rand(10, 2)
  P2 = np.random.rand(20, 2)
  scale = 1.

  indH3, valH3 = create_tensor_3rd_random_fc(P1, P2, scale)
  assert indH3.shape[0] == valH3.shape[0]

  indH3, valH3 = create_tensor_3rd_tri_tri(P1, P2, scale)
  assert indH3.shape[0] == valH3.shape[0]

  indH3, valH3 = create_tensor_3rd_tri_fc(P1, P2, scale)
  assert indH3.shape[0] == valH3.shape[0]

  indH3, valH3 = create_tensor_3rd_tri_random_fc(P1, P2, scale)
  assert indH3.shape[0] == valH3.shape[0]

  indH3, valH3 = create_tensor_3rd(P1, P2, scale)
  assert indH3.shape[0] == valH3.shape[0]


def test_create_tensor_3rd_scale_none():
  P1 = np.random.rand(10, 2)
  P2 = np.random.rand(20, 2)
  scale = None

  indH3, valH3 = create_tensor_3rd_random_fc(P1, P2)
  assert indH3.shape[0] == valH3.shape[0]
  indH3, valH3 = create_tensor_3rd_random_fc(P1, P2, scale=scale)
  assert indH3.shape[0] == valH3.shape[0]

  indH3, valH3 = create_tensor_3rd_tri_tri(P1, P2)
  assert indH3.shape[0] == valH3.shape[0]
  indH3, valH3 = create_tensor_3rd_tri_tri(P1, P2, scale=scale)
  assert indH3.shape[0] == valH3.shape[0]

  indH3, valH3 = create_tensor_3rd_tri_fc(P1, P2)
  assert indH3.shape[0] == valH3.shape[0]
  indH3, valH3 = create_tensor_3rd_tri_fc(P1, P2, scale=scale)
  assert indH3.shape[0] == valH3.shape[0]

  indH3, valH3 = create_tensor_3rd_tri_random_fc(P1, P2)
  assert indH3.shape[0] == valH3.shape[0]
  indH3, valH3 = create_tensor_3rd_tri_random_fc(P1, P2, scale=scale)
  assert indH3.shape[0] == valH3.shape[0]

  indH3, valH3 = create_tensor_3rd(P1, P2)
  assert indH3.shape[0] == valH3.shape[0]
  indH3, valH3 = create_tensor_3rd(P1, P2, scale=scale)
  assert indH3.shape[0] == valH3.shape[0]


def _test_create_tensor_3rd_rng_helper(func, P1, P2, scale, seed):
  rng1 = np.random.default_rng(seed)
  rng2 = np.random.default_rng(seed)

  for _ in range(10):
    ind1, val1 = func(P1, P2, scale, rng=rng1)
    ind2, val2 = func(P1, P2, scale, rng=rng2)
    np.testing.assert_allclose(ind1, ind2)
    np.testing.assert_allclose(val1, val2)


def test_create_tensor_3rd_rng():
  rng = np.random.default_rng(1234)
  P1 = rng.random(size=(10, 2))
  P2 = rng.random(size=(20, 2))
  scale = 1.

  _test_create_tensor_3rd_rng_helper(
      create_tensor_3rd_random_fc,
      P1, P2, scale,
      seed=1024
  )

  _test_create_tensor_3rd_rng_helper(
      create_tensor_3rd_tri_random_fc,
      P1, P2, scale,
      seed=1024
  )

  _test_create_tensor_3rd_rng_helper(
      create_tensor_3rd,
      P1, P2, scale,
      seed=1024
  )


if __name__ == '__main__':
  test_distances_to_affinity_scores()
  test_sort_each_edge_1st()
  test_sort_each_edge_2nd()
  test_sort_each_edge_3rd()
  test_unique_all_edges_1st()
  test_unique_all_edges_2nd()
  test_unique_all_edges_3rd()
  test_unique_all_edges_3rd_directed()
  test_select_edges_with_different_nodes()
  test_select_edges_with_different_nodes_2()
  test_select_edges_with_different_nodes_sorted()
  test_select_edges_with_different_nodes_sorted_2()
  test_sort_filter_unique()
  test_sort_filter_unique_2()
  test_indH3()
  test_valH3()
  test_create_feature_tensor_indH()
  test_create_feature_tensor_T1_T2()
  test_create_tensor_2nd()
  test_create_tensor_3rd()
  test_create_tensor_3rd_scale_none()
  test_create_tensor_3rd_rng()
