import abc
import itertools
import collections.abc

import scipy
import numpy as np
import tensorflow as tf

from utils.create_tensor import create_feature_tensor

from .. import hypergraph
from ..hypergraph import HypergraphsTuple
from ..hypergraph_utils_tf import set_zero_feature
from ..hypergraph_utils_tf import merge_graphs
from ..hypergraph_utils_tf import concat_attributes


def _set_initial_nodes(graph, dtype=tf.float32):
  n_all_node = tf.reduce_sum(graph.n_node)
  nodes = tf.ones([n_all_node], dtype=dtype)
  cols = tf.cast(graph.n_col, dtype=dtype)
  nodes = nodes / tf.repeat(cols, repeats=graph.n_node, axis=0)
  nodes = tf.reshape(nodes, [-1, 1])
  graph = graph.replace(nodes=nodes)
  return graph


def _normalize_edge(edges):
  # edges, _ = tf.linalg.normalize(edges)
  edges = edges / tf.reduce_max(edges)
  # mine = tf.reduce_min(edges)
  # maxe = tf.reduce_max(edges)
  # edges = (edges - mine) / (maxe - mine)
  # edges = edges - tf.reduce_mean(edges) / tf.math.reduce_std(edges)

  return edges


def _undirected_to_partially_directed(indH, valH):
  """set of senders to one receiver"""
  N = indH.shape[1]

  indH_list = [indH]
  for shift in range(1, N):
    indH_list.append(np.roll(indH, shift, axis=1))
  indH = np.concatenate(indH_list, axis=0)

  valH = np.repeat(valH, repeats=N, axis=0)

  return indH, valH


def _undirected_to_fully_directed(indH, valH):
  order = indH.shape[-1]
  indices = np.asarray(list(itertools.permutations(range(order))))
  indH = indH[:, indices].reshape(-1, order)
  N = len(indices)
  valH = np.repeat(valH, repeats=N, axis=0)
  return indH, valH


def _undirected_to_directed(indH, valH, is_senders_ordered=True):
  if is_senders_ordered:
    return _undirected_to_fully_directed(indH, valH)
  else:
    return _undirected_to_partially_directed(indH, valH)


def _normalize_point_set(P):
  P = P - np.mean(P, axis=0, keepdims=True)
  distance = scipy.spatial.distance.cdist(P, P)
  P = P / np.max(distance)
  return P


class BaseProblemGenerator(abc.ABC):
  def __init__(self, rng=None):
    self.default_rng = np.random.default_rng(rng)

  @abc.abstractmethod
  def generate_problem(self, settings, rng=None):
    pass

  def generate_problems(self, batch_size, settings, rng=None):
    if rng is None:
      rng = self.default_rng
    rng = np.random.default_rng(rng)

    settings_template = settings
    problems = []
    for _ in range(batch_size):
      settings = settings_template.copy()
      for key in settings_template.keys():
        value = settings_template[key]
        if isinstance(value, collections.abc.Sequence) and not isinstance(value, str):
          settings[key] = rng.choice(value)

      problem = self.generate_problem(settings, rng=rng)
      problems.append(problem)
    return problems


class GraphCreator:
  def __init__(self, cfg):
    self.input_dtype = cfg.INPUT_DTYPE
    self.input_shape = cfg.INPUT_SHAPE
    self.input_signature = cfg.INPUT_SIGNATURE

    self.target_dtype = cfg.TARGET_DTYPE
    self.target_shape = cfg.TARGET_SHAPE
    self.target_signature = cfg.TARGET_SIGNATURE

    # TODO: Is it right to define `repeat_hyperedge` here?
    self.default_repeat_hyperedge = cfg.REPEAT_HYPEREDGE

    self.default_normalize_point_set = cfg.NORMALIZE_POINT_SET

  def get_input_graph(self, problem):
    """
    Value of `repeat_hyperedge` should be one of ['same', 'roll', 'permutation']
    * 'same': same as the input;
    * 'roll': roll the input;
    * 'permutation': permutate the input.
    """
    repeat_hyperedge = self.default_repeat_hyperedge
    normalize_point_set = self.default_normalize_point_set

    assert repeat_hyperedge in ['same', 'roll', 'permutation']

    P1 = problem["P1"]
    P2 = problem["P2"]
    nP1 = problem['nP1']
    nP2 = problem['nP2']
    indH = problem['indH3']
    valH = problem['valH3']

    if normalize_point_set:
      P1 = _normalize_point_set(P1)
      P2 = _normalize_point_set(P2)

    if repeat_hyperedge != 'same':
      is_senders_ordered = (repeat_hyperedge == 'permutation')
      indH, valH = _undirected_to_directed(indH, valH, is_senders_ordered=is_senders_ordered)

    hyperedges = tf.convert_to_tensor(indH, dtype=tf.int32)

    nodes = create_feature_tensor(P1, P2, indH=np.arange(nP1 * nP2).reshape([-1, 1]))
    nodes = tf.convert_to_tensor(nodes, dtype=tf.float32)
    n_node = tf.convert_to_tensor([nP1 * nP2], dtype=tf.int32)

    # edges = tf.reshape(tf.convert_to_tensor(valH, dtype=tf.float32), [-1, 1])
    edges = create_feature_tensor(P1, P2, indH=indH)
    edges = tf.convert_to_tensor(edges, dtype=tf.float32)
    # n_edge = indH.shape[0]
    n_edge = tf.convert_to_tensor([hyperedges.shape[0]], dtype=tf.int32)

    n_row = tf.convert_to_tensor([nP1], dtype=tf.int32)
    n_col = tf.convert_to_tensor([nP2], dtype=tf.int32)

    nrow = n_row[0]
    ncol = n_col[0]
    row_id = tf.repeat(tf.range(nrow), repeats=ncol, axis=0)
    col_id = tf.tile(tf.range(ncol), multiples=[nrow])

    n_global = tf.convert_to_tensor([1], dtype=tf.int32)

    graph = HypergraphsTuple(nodes=nodes, n_node=n_node,
                             edges=edges, n_edge=n_edge, hyperedges=hyperedges,
                             rows=None, n_row=n_row, row_id=row_id,
                             cols=None, n_col=n_col, col_id=col_id,
                             globals=None, n_global=n_global)

    # graph = graph.replace(edges=_normalize_edge(graph.edges))
    if graph.nodes is None:
      graph = _set_initial_nodes(graph)
    # graph = set_zero_feature(graph, node_size=1, global_size=1)
    # graph = set_zero_feature(graph, global_size=1, row_size=1, col_size=1)
    size_dict = {}
    for k in hypergraph.HYPERGRAPH_FEATURE_FIELDS:
      if getattr(graph, k) is None:
        size_dict[k] = getattr(HypergraphsTuple(*self.input_shape), k)[1:]
    graph = set_zero_feature(graph, size_dict)

    return graph

  def get_target_graph(self, input_graph, assignment_matrix):
    assignment_matrix = tf.cast(assignment_matrix, tf.float32)  # TODO: user-specific dtype
    true_node = tf.reshape(assignment_matrix, [-1, 1])
    return input_graph.replace(nodes=true_node)

  def create_data(self, problem_list):
    input_graph_list = []
    target_graph_list = []
    for problem in problem_list:
      input_graph = self.get_input_graph(problem)
      target_graph = self.get_target_graph(input_graph, problem["assignmentMatrix"])
      input_graph_list.append(input_graph)
      target_graph_list.append(target_graph)
    inputs = merge_graphs(input_graph_list)

    # NOTE: 1
    # targets = merge_graphs(target_graph_list)

    # NOTE: 2
    # target_nodes_list = [g.nodes for g in target_graph_list]
    # target_nodes = tf.concat(target_nodes_list, axis=0)
    # targets = inputs.replace(nodes=target_nodes)

    # NOTE: 3
    key = hypergraph.NODES
    target_nodes = concat_attributes(target_graph_list, keys=[key], axis=0)[key]
    targets = inputs.replace(nodes=target_nodes)

    return inputs, targets


class Dataset:
  def __init__(self, problem_generator, graph_creator):
    self.input_dtype = graph_creator.input_dtype
    self.input_shape = graph_creator.input_shape

    self.target_dtype = graph_creator.target_dtype
    self.target_shape = graph_creator.target_shape

    self.problem_generator = problem_generator
    self.graph_creator = graph_creator

  def generate_data(self, batch_size, settings):
    problem_list = self.problem_generator.generate_problems(batch_size, settings)
    return self.graph_creator.create_data(problem_list)


def get_dataloader(dataset, num_batch, batch_size, settings, num_parallel_calls=None, deterministic=None):
  def wrap_create_data():
    inputs, targets = dataset.generate_data(batch_size, settings)
    return inputs + targets

  def g(x):
    data = tf.py_function(wrap_create_data, [], dataset.input_dtype + dataset.target_dtype)

    N = len(dataset.input_dtype)
    inputs = data[:N]
    targets = data[N:]

    for tensor, shape in zip(inputs, dataset.input_shape):
      tensor.set_shape(shape)

    for tensor, shape in zip(targets, dataset.target_shape):
      tensor.set_shape(shape)

    return HypergraphsTuple(*inputs), HypergraphsTuple(*targets)

  ds = tf.data.Dataset.range(num_batch).map(g,
                                            num_parallel_calls=num_parallel_calls,
                                            deterministic=deterministic)

  return ds
