import random
import numpy as np
import tensorflow as tf

from HNN_HM.hnn_hm import hypergraph
from HNN_HM.hnn_hm.config.utils import get_config
from HNN_HM.hnn_hm.data.data import get_dataloader
from HNN_HM.hnn_hm.data.utils import get_dataset


DATASET_NAMES = [
    "Synthetic",
    "House",
    "Willow",
    "Pascal VOC",
    "SPair-71k"
]


def set_random_seed(seed):
  random.seed(seed)
  np.random.seed(seed)
  tf.random.set_seed(seed)


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


def assert_same_dataset(dataset1, dataset2, max_num=None):
  cnt = 0
  for (input_graph1, target_graph1), (input_graph2, target_graph2) in zip(dataset1, dataset2):
    cnt += 1
    print(f'Case # {cnt}')
    assert_same_HypergraphsTuple(input_graph1, input_graph2)
    assert_same_HypergraphsTuple(target_graph1, target_graph2)
    if max_num is not None and cnt >= max_num:
        break


def get_batch_generator(dataset, num_batch, batch_size, settings):
  for _ in range(num_batch):
    inputs, targets = dataset.generate_data(batch_size, settings)
    yield inputs, targets


def get_train_and_val_datasets_simple(dataset_name):
  cfg = get_config(dataset_name=dataset_name)

  _dataset_tr = get_dataset(sets="train", cfg=cfg, rng=cfg.RNG_TR)
  dataset_tr = get_batch_generator(_dataset_tr,
                                   cfg.NUM_ITERATIONS_TR,
                                   cfg.BATCH_SIZE_TR,
                                   cfg.SETTINGS_TR)

  _dataset_ge = get_dataset(sets="val", cfg=cfg, rng=cfg.RNG_GE)
  dataset_ge = get_batch_generator(_dataset_ge,
                                   cfg.NUM_ITERATIONS_GE,
                                   cfg.BATCH_SIZE_GE,
                                   cfg.SETTINGS_GE)

  return dataset_tr, dataset_ge


# def get_train_and_val_datasets(dataset_name):
#   cfg = get_config(dataset_name=dataset_name)

#   _dataset_tr = get_dataset(sets="train", cfg=cfg, rng=cfg.RNG_TR)
#   dataset_tr = get_dataloader(_dataset_tr,
#                               cfg.NUM_ITERATIONS_TR,
#                               cfg.BATCH_SIZE_TR,
#                               cfg.SETTINGS_TR,
#                               num_parallel_calls=cfg.NUM_PARALLEL_CALLS,
#                               deterministic=cfg.DETERMINISTIC_TR)
#   if getattr(cfg, 'DATASET_CACHE_TR', None) is not None:
#       dataset_tr = dataset_tr.cache(cfg.DATASET_CACHE_TR)
#   dataset_tr = dataset_tr.prefetch(cfg.PREFETCH_TR)

#   _dataset_ge = get_dataset(sets="val", cfg=cfg, rng=cfg.RNG_GE)
#   dataset_ge = get_dataloader(_dataset_ge,
#                               cfg.NUM_ITERATIONS_GE,
#                               cfg.BATCH_SIZE_GE,
#                               cfg.SETTINGS_GE,
#                               num_parallel_calls=cfg.NUM_PARALLEL_CALLS,
#                               deterministic=cfg.DETERMINISTIC_GE)
#   if getattr(cfg, 'DATASET_CACHE_GE', None) is not None:
#       dataset_ge = dataset_ge.cache(cfg.DATASET_CACHE_GE)
#   dataset_ge = dataset_ge.prefetch(cfg.PREFETCH_GE)

#   return dataset_tr, dataset_ge


def get_train_and_val_datasets(dataset_name):
  cfg = get_config(dataset_name=dataset_name)

  _dataset_tr = get_dataset(sets="train", cfg=cfg, rng=cfg.RNG_TR)
  dataset_tr = get_dataloader(_dataset_tr,
                              cfg.NUM_ITERATIONS_TR,
                              cfg.BATCH_SIZE_TR,
                              cfg.SETTINGS_TR,
                              num_parallel_calls=None,
                              deterministic=True)
  if getattr(cfg, 'DATASET_CACHE_TR', None) is not None:
      dataset_tr = dataset_tr.cache(cfg.DATASET_CACHE_TR)
  dataset_tr = dataset_tr.prefetch(cfg.PREFETCH_TR)

  _dataset_ge = get_dataset(sets="val", cfg=cfg, rng=cfg.RNG_GE)
  dataset_ge = get_dataloader(_dataset_ge,
                              cfg.NUM_ITERATIONS_GE,
                              cfg.BATCH_SIZE_GE,
                              cfg.SETTINGS_GE,
                              num_parallel_calls=None,
                              deterministic=True)
  if getattr(cfg, 'DATASET_CACHE_GE', None) is not None:
      dataset_ge = dataset_ge.cache(cfg.DATASET_CACHE_GE)
  dataset_ge = dataset_ge.prefetch(cfg.PREFETCH_GE)

  return dataset_tr, dataset_ge


def _test_data_helper(get_train_and_val_datasets_func, max_num):
  set_random_seed(1234)
  for dataset_name in DATASET_NAMES:
    print(f'Dataset: {dataset_name}')
    dataset_tr1, dataset_ge1 = get_train_and_val_datasets_func(dataset_name)
    dataset_tr2, dataset_ge2 = get_train_and_val_datasets_func(dataset_name)
    print(f'Dataset: {dataset_name} trainin set')
    assert_same_dataset(dataset_tr1, dataset_tr2, max_num=max_num)
    print(f'Dataset: {dataset_name} validation set')
    assert_same_dataset(dataset_ge1, dataset_ge2, max_num=max_num)
    print(f'Dataset: {dataset_name} is OK!')


def test_get_dataset():
  _test_data_helper(get_train_and_val_datasets_simple, max_num=10)


def test_get_dataloader():
  _test_data_helper(get_train_and_val_datasets, max_num=10)


if __name__ == '__main__':
  test_get_dataset()
  test_get_dataloader()
