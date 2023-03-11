import tensorflow as tf

from . import hypergraph
from .hypergraph import HypergraphsTuple


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


def hypergraphs_tuple_to_dict(graph):
  return {k: getattr(graph, k) for k in hypergraph.ALL_FIELDS}


def dict_to_hypergraphs_tuple(graph_dict):
  return HypergraphsTuple(**graph_dict)


def set_zero_feature(graph,
                     feature_shape_dict,
                     dtype=tf.float32):
  for field, feature_shape in feature_shape_dict.items():
    idx = hypergraph.HYPERGRAPH_FEATURE_FIELDS.index(field)
    n_feat = getattr(graph, hypergraph.HYPERGRAPH_NUMBER_FIELDS[idx])
    n_feat = tf.reduce_sum(n_feat)
    shape = (n_feat,) + feature_shape
    feat = tf.zeros(shape, dtype=dtype)
    graph = graph.replace(**{field: feat})
  return graph


def none_or_concat(values, axis):
  """
  values: list of tensors or None (If none exists, all values should be None)
  axis:   axis to concat all tensors
  """
  # Count the number of None
  cnt = sum(t is None for t in values)
  if cnt > 0:
    assert cnt == len(values), "All values should be None if None exists"
    return None
  return tf.concat(values, axis=axis)


def concat_attributes(graphs, keys, axis):
  out = {}
  for k in keys:
    value = none_or_concat([getattr(g, k) for g in graphs], axis=axis)
    out[k] = value
  return out


def concat_graphs(graphs, basic_graph_index=0):
  features = concat_attributes(graphs, hypergraph.HYPERGRAPH_FEATURE_FIELDS, axis=1)
  return graphs[basic_graph_index].replace(**features)


def merge_graphs(graph_list):
  assert len(graph_list) > 0

  keys = hypergraph.HYPERGRAPH_FEATURE_FIELDS + hypergraph.HYPERGRAPH_NUMBER_FIELDS
  attributes = concat_attributes(graph_list, keys, axis=0)

  # TODO: Right? Need more check!
  for index, number, is_index_of_nodes in zip(hypergraph.HYPEREDGES_INDEX_FIELDS,
                                              hypergraph.HYPEREDGES_NUMBER_FIELDS,
                                              hypergraph.HYPEREDGES_INDEX_OF_NODES):
    hyperedges_list = []
    cnt = 0
    for graph in graph_list:
      hyperedges = getattr(graph, index) + cnt
      hyperedges_list.append(hyperedges)
      if is_index_of_nodes:
          cnt += tf.reduce_sum(getattr(graph, hypergraph.N_NODE))
      else:
          cnt += tf.reduce_sum(getattr(graph, number))
    hyperedges = none_or_concat(hyperedges_list, axis=0)
    attributes[index] = hyperedges

  return HypergraphsTuple(**attributes)
