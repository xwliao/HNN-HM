import os
import numpy as np
import tensorflow as tf
from .. import hypergraph
from ..hypergraph import HypergraphsTuple


class BaseConfig:
  def __init__(self):
    # ==================================

    self.DATASET_NAME = "Unknown"  # TODO: **REQUIRED**

    # ==================================

    self.REPEAT_HYPEREDGE = 'same'
    # self.REPEAT_HYPEREDGE = 'permutation'

    self.FORCE_SYMMETRY = True

    # Number of processing (message-passing) steps.
    self.NUM_PROCESSING_STEPS = 10
    self.NODE_OUTPUT_SIZE = 1

    # self.PROJECTION_ITER = 100
    # self.PROJECTION_ITER = 10

    # ==================================

    node_feature_dim = 4
    edge_feature_dim = 12

    input_info = HypergraphsTuple(**{
        hypergraph.NODES: ((None, node_feature_dim), 'float32'),
        hypergraph.N_NODE: ((None,), 'int32'),
        hypergraph.EDGES: ((None, edge_feature_dim), 'float32'),
        hypergraph.N_EDGE: ((None,), 'int32'),
        hypergraph.HYPEREDGES: ((None, 3), 'int32'),
        hypergraph.ROWS: ((None, 1), 'float32'),
        hypergraph.N_ROW: ((None,), 'int32'),
        hypergraph.ROW_ID: ((None,), 'int32'),
        hypergraph.COLS: ((None, 1), 'float32'),
        hypergraph.N_COL: ((None,), 'int32'),
        hypergraph.COL_ID: ((None,), 'int32'),
        hypergraph.GLOBALS: ((None, 1), 'float32'),
        hypergraph.N_GLOBAL: ((None,), 'int32'),
    })
    self.INPUT_SHAPE = tuple(shape for (shape, _) in input_info)
    self.INPUT_DTYPE = tuple(dtype for (_, dtype) in input_info)
    self.INPUT_SIGNATURE = HypergraphsTuple(*(tf.TensorSpec(shape=shape, dtype=dtype)
                                              for (shape, dtype) in input_info))

    target_info = input_info.replace(nodes=((None, 1), 'float32'))
    self.TARGET_SHAPE = tuple(shape for (shape, _) in target_info)
    self.TARGET_DTYPE = tuple(dtype for (_, dtype) in target_info)
    self.TARGET_SIGNATURE = HypergraphsTuple(*(tf.TensorSpec(shape=shape, dtype=dtype)
                                               for (shape, dtype) in target_info))
    # ==================================

    self.NORMALIZE_POINT_SET = False  # NOTE: pay attention

    # ==================================
    # Random

    self.GLOBAL_RANDOM_SEED = 1

    ss = np.random.SeedSequence(654321)
    seeds = ss.spawn(2)
    self.RNG_TR = np.random.default_rng(seeds[0])
    self.RNG_GE = np.random.default_rng(seeds[1])

    # ==================================

    self.NUM_PARALLEL_CALLS = tf.data.experimental.AUTOTUNE

    # ==================================
    # Checkpoint and log

    self.DEFAULT_RESULT_DIR_ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "results")

    # Note: self.LOG_DIR and self.CHECKPOINT_DIR and be set by using `self.set_result_dir`
    self.LOG_DIR = "logs"                 # TODO: **REQUIRED**
    self.CHECKPOINT_DIR = "checkpoints"   # TODO: **REQUIRED**

    self.LOG_EVERY_ITERATIONS = 500

    # How much time between logging and printing the current results.
    # self.LOG_EVERY_SECONDS = 120
    # self.LOG_EVERY_SECONDS = 600

    # ==================================
    # For training

    self.NUM_ITERATIONS_TR = 300000  # TODO: **REQUIRED**
    self.BATCH_SIZE_TR = 16          # TODO: **REQUIRED**

    # Note:
    # Value of DATASET_CACHE_TR and DATASET_CACHE_GE can be:
    #   <undefined>: same as None
    #   None: do nothing, so each iteration of the dataset is different
    #   '': cache in memory
    #   <PATH_TO_FILE>: cache to a file
    self.DATASET_CACHE_TR = None
    self.PREFETCH_TR = tf.data.experimental.AUTOTUNE
    self.DETERMINISTIC_TR = True

    # ==================================
    # For validation

    self.NUM_ITERATIONS_GE = 50
    self.BATCH_SIZE_GE = 2

    # self.DATASET_CACHE_GE = None
    self.DATASET_CACHE_GE = ''
    self.PREFETCH_GE = tf.data.experimental.AUTOTUNE
    self.DETERMINISTIC_GE = True

    # ==================================
    # Optimizer

    # ADAM
    self.OPTIMIZER = 'adam'
    self.AMSGRAD = True

    # TODO: **REQUIRED**
    # self.LEARNING_RATE = 1e-3
    self.LEARNING_RATE = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=1e-3,
                                                                        decay_steps=10000,
                                                                        decay_rate=0.96,
                                                                        staircase=True)

    # SGD
    # self.OPTIMIZER = 'sgd'
    # self.LEARNING_RATE = 0.1
    # self.MOMENTUM = 0.9

    # ==================================

    self.SETTINGS_TR = {}      # TODO: **REQUIRED**
    self.SETTINGS_GE = {}      # TODO: **REQUIRED**
    self.SETTINGS_SIMPLE = {}  # TODO: **REQUIRED**

  def set_result_dir(self, result_dir=None, name=None):
    if result_dir is None:
      assert name is not None
      result_dir = os.path.join(self.DEFAULT_RESULT_DIR_ROOT, name)

    self.LOG_DIR = os.path.join(result_dir, 'logs')
    self.CHECKPOINT_DIR = os.path.join(result_dir, 'checkpoints')
