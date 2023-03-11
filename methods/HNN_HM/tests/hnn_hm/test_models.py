import numpy as np
import tensorflow as tf

from HNN_HM.hnn_hm import hypergraph
from HNN_HM.hnn_hm.hypergraph import HypergraphsTuple
from HNN_HM.hnn_hm.hypergraph_utils_tf import set_zero_feature
from HNN_HM.hnn_hm.hypergraph_utils_tf import merge_graphs
from HNN_HM.hnn_hm.models import HypergraphModel


def debug_test_repeat(arr, n, n_repeats):
  assert(arr.shape[0] == n * n_repeats)
  a = arr[:n]
  for k in range(n_repeats):
    b = arr[k * n:(k + 1) * n]
    print('Repeat {} of {}'.format(k + 1, n_repeats))
    print(a - b)
    np.testing.assert_allclose(b, a)


def tf_print(name, data):
  # TODO: just for debug
  if data is None:
    tf.print(name, 'None')
  else:
    tf.print(name, data.dtype, 'shape:', tf.shape(data))
    tf.print(data)


def debug_print_HypergraphsTuple(graph):
  # TODO: just for debug
  for k in hypergraph.ALL_FIELDS:
    tf_print(k, getattr(graph, k))


def assert_none_or_equal(x, y, *args, **kwargs):
  if x is None or y is None:
    assert (x is None and y is None)
  else:
    tf.debugging.assert_equal(x, y, *args, **kwargs)


def assert_none_or_near(x, y, *args, **kwargs):
  if x is None or y is None:
    assert (x is None and y is None)
  else:
    tf.debugging.assert_near(x, y, *args, **kwargs)


def assert_same_HypergraphsTuple(graph1, graph2, rtol=None, atol=None):
  for k in hypergraph.HYPERGRAPH_INDEX_FIELDS:
    assert_none_or_equal(getattr(graph1, k), getattr(graph2, k))

  for k in hypergraph.HYPERGRAPH_NUMBER_FIELDS:
    assert_none_or_equal(getattr(graph1, k), getattr(graph2, k))

  for k in hypergraph.HYPERGRAPH_FEATURE_FIELDS:
    assert_none_or_near(getattr(graph1, k), getattr(graph2, k), rtol=rtol, atol=atol)

  # for edges1, edges2 in zip(graph1.hyperedges, graph2.hyperedges):
  #   assert_none_or_equal(tf.reduce_all(tf.equal(edges1, edges2)), True)


def generate_random_graph(edge_feature_dim):
  nP1 = 2
  nP2 = 2

  nodes = tf.random.uniform((nP1 * nP2, 1), dtype=tf.float32)
  n_node = tf.reshape(tf.shape(nodes)[0], [-1])

  edges = tf.random.uniform((2, edge_feature_dim), dtype=tf.float32)
  n_edge = tf.reshape(tf.shape(edges)[0], [-1])
  hyperedges = tf.constant([[0, 1, 2],
                            [1, 2, 3]], dtype=tf.int32)

  n_row = tf.convert_to_tensor([nP1], dtype=tf.int32)
  n_col = tf.convert_to_tensor([nP2], dtype=tf.int32)

  nrow = n_row[0]
  ncol = n_col[0]
  row_id = tf.repeat(tf.range(nrow), repeats=ncol, axis=0)
  col_id = tf.tile(tf.range(ncol), multiples=[nrow])

  globals = tf.random.uniform((1, 1), dtype=tf.float32)
  n_global = tf.convert_to_tensor([1], dtype=tf.int32)

  graph = HypergraphsTuple(nodes=nodes,
                           n_node=n_node,
                           edges=edges,
                           n_edge=n_edge,
                           hyperedges=hyperedges,
                           rows=None,
                           n_row=n_row,
                           row_id=row_id,
                           cols=None,
                           n_col=n_col,
                           col_id=col_id,
                           globals=globals,
                           n_global=n_global)
  graph = set_zero_feature(graph, dict(rows=(1,), cols=(1,)))

  debug_print_HypergraphsTuple(graph)  # TODO

  return graph


def get_repeated_graph_batch(batch_size, edge_feature_dim):
  nP1 = 2
  nP2 = 2

  nodes = tf.constant([[-1.], [-0.5], [0.5], [1.]], dtype=tf.float32)
  n_node = tf.constant([nodes.shape[0]], dtype=tf.int32)

  edges = tf.constant([[0.5] * edge_feature_dim,
                       [1.0] * edge_feature_dim], dtype=tf.float32)
  n_edge = tf.constant([edges.shape[0]], dtype=tf.int32)
  hyperedges = tf.constant([[0, 1, 2],
                            [1, 2, 3]], dtype=tf.int32)

  n_row = tf.convert_to_tensor([nP1], dtype=tf.int32)
  n_col = tf.convert_to_tensor([nP2], dtype=tf.int32)

  nrow = n_row[0]
  ncol = n_col[0]
  row_id = tf.repeat(tf.range(nrow), repeats=ncol, axis=0)
  col_id = tf.tile(tf.range(ncol), multiples=[nrow])

  globals = tf.constant([[1.]], dtype=tf.float32)
  n_global = tf.convert_to_tensor([1], dtype=tf.int32)

  graph = HypergraphsTuple(nodes=nodes,
                           n_node=n_node,
                           edges=edges,
                           n_edge=n_edge,
                           hyperedges=hyperedges,
                           rows=None,
                           n_row=n_row,
                           row_id=row_id,
                           cols=None,
                           n_col=n_col,
                           col_id=col_id,
                           globals=globals,
                           n_global=n_global)
  graph = set_zero_feature(graph, dict(rows=(1,), cols=(1,)))

  debug_print_HypergraphsTuple(graph)  # TODO

  graphs = merge_graphs([graph] * batch_size)

  debug_print_HypergraphsTuple(graphs)  # TODO

  return graphs


def _test_HypergraphModel_helper(edge_feature_dim, force_symmetry):
  graph = generate_random_graph(edge_feature_dim=edge_feature_dim)

  model = HypergraphModel(num_processing_steps=10,
                          node_output_size=1,
                          force_symmetry=force_symmetry)

  out_graphs = model(tuple(graph))

  model.summary()

  for out_graph in out_graphs:
    assert_none_or_equal(out_graph.nodes.shape, [graph.nodes.shape[0], 1])
    assert_none_or_equal(out_graph.hyperedges, graph.hyperedges)
    assert_none_or_equal(out_graph.n_node, graph.n_node)
    assert_none_or_equal(out_graph.n_edge, graph.n_edge)


def test_HypergraphModel():
  _test_HypergraphModel_helper(edge_feature_dim=1, force_symmetry=False)
  _test_HypergraphModel_helper(edge_feature_dim=12, force_symmetry=True)


def _test_HypergraphModel_device_helper(device1,
                                        device2,
                                        edge_feature_dim,
                                        force_symmetry):
  graph = generate_random_graph(edge_feature_dim=edge_feature_dim)
  graph = tuple(graph)

  model = HypergraphModel(num_processing_steps=10,
                          node_output_size=1,
                          force_symmetry=force_symmetry)

  with tf.device(device1):
    out_graphs_cpu = model(graph)

  with tf.device(device2):
    out_graphs_gpu = model(graph)

  # TODO
  rtol = None
  # atol = None
  atol = 1e-4

  for out_graph_cpu, out_graph_gpu in zip(out_graphs_cpu, out_graphs_gpu):
    # TODO:
    print('=' * 10)
    debug_print_HypergraphsTuple(out_graph_cpu)
    print('-' * 10)
    debug_print_HypergraphsTuple(out_graph_gpu)
    print('=' * 10)
    assert_same_HypergraphsTuple(out_graph_gpu, out_graph_cpu, rtol=rtol, atol=atol)


def test_HypergraphModel_device():
  devices = ["cpu", "gpu"]
  for device1 in devices:
    for device2 in devices:
      _test_HypergraphModel_device_helper(device1=device1,
                                          device2=device2,
                                          edge_feature_dim=1,
                                          force_symmetry=False)
      _test_HypergraphModel_device_helper(device1=device1,
                                          device2=device2,
                                          edge_feature_dim=12,
                                          force_symmetry=True)


def _test_HypergraphModel_batch_helper(edge_feature_dim, force_symmetry):
  model = HypergraphModel(num_processing_steps=10,
                          node_output_size=1,
                          force_symmetry=force_symmetry)

  graph1 = get_repeated_graph_batch(batch_size=1, edge_feature_dim=edge_feature_dim)
  out_graph1 = model(tuple(graph1))[-1]
  out_graph1_again = model(tuple(graph1))[-1]

  print("=" * 30)
  print("input1:")
  debug_print_HypergraphsTuple(graph1)
  print("-" * 30)
  print("output1:")
  debug_print_HypergraphsTuple(out_graph1)
  print("-" * 30)
  print("output1_again:")
  debug_print_HypergraphsTuple(out_graph1_again)
  print("=" * 30)

  # TODO
  # n2 = 2
  n2 = 10
  graph2 = get_repeated_graph_batch(batch_size=n2, edge_feature_dim=edge_feature_dim)
  out_graph2 = model(tuple(graph2))[-1]

  print("=" * 30)
  print("input2:")
  debug_print_HypergraphsTuple(graph2)
  print("-" * 30)
  print("output2:")
  debug_print_HypergraphsTuple(out_graph2)
  print("=" * 30)

  # TODO
  rtol = None
  # atol = None
  atol = 1e-4

  N = graph1.nodes.shape[0]
  assert_none_or_equal(out_graph1.nodes.shape, [N, 1])
  assert_none_or_near(out_graph1_again.nodes, out_graph1.nodes, rtol=rtol, atol=atol)
  # debug_test_repeat(out_graph2.nodes.numpy(), N, n2)
  assert_none_or_near(out_graph2.nodes, tf.tile(out_graph2.nodes[:N], [n2, 1]), rtol=rtol, atol=atol)
  assert_none_or_near(out_graph2.nodes[:N], out_graph1.nodes, rtol=rtol, atol=atol)


def test_HypergraphModel_batch():
  _test_HypergraphModel_batch_helper(edge_feature_dim=1, force_symmetry=False)
  _test_HypergraphModel_batch_helper(edge_feature_dim=12, force_symmetry=True)


if __name__ == '__main__':
  test_HypergraphModel()
  test_HypergraphModel_device()
  test_HypergraphModel_batch()
