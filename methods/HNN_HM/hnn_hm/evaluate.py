import os

import tensorflow as tf

from .models import HypergraphModel
from .data import utils as data_utils


def _get_predict_function_helper(model, graph_creator, device=None):
    @tf.function(input_signature=[graph_creator.input_signature])
    def tf_function_predict(graph):
      return model(graph)[-1]

    def tf_predict(problem):
      nP1 = problem['nP1']
      nP2 = problem['nP2']
      input = graph_creator.get_input_graph(problem)
      output = tf_function_predict(input)
      X = tf.reshape(output.nodes, [nP1, nP2])
      return X

    def predict_with_device(problem):
      if device is None:
          X = tf_predict(problem)
      else:
        with tf.device(device):
          X = tf_predict(problem)
      return X.numpy()

    return predict_with_device


def get_predict_function(cfg=None, dataset_name=None,
                         checkpoint_path=None, checkpoint_name=None, checkpoint_dir=None,
                         compiled=True, device=None):
  """
  checkpoint_path: the full path to the checkpoint (usually end with ".ckpt")
  checkpoint_name: the name of the checkpoint under the checkpoint directory (usually end with ".ckpt")
  checkpoint_dir: the checkpoint directory (if not provied, use the default checkpoint directory)
  compiled: whether to compile the model (can speed up the first call if compiled)

  Note:
  * If both the `checkpoint_path` parameter and the `checkpoint_name` parameter are provied,
    the `checkpoint_path` parameter will be used.
  * If neither the `checkpoint_path` parameter nor the `checkpoint_name` parameter is provied,
    the latest checkpoint under the checkpoint directory will be used.
  """
  if cfg is None:
    assert dataset_name is not None
    from .config.utils import get_config
    cfg = get_config(dataset_name=dataset_name)

  if checkpoint_path is None:
    if checkpoint_dir is None:
      checkpoint_dir = cfg.CHECKPOINT_DIR
    assert os.path.isdir(checkpoint_dir), "Can't find the checkpoint directory: {}".format(checkpoint_dir)

    if checkpoint_name is not None:
      checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
    else:
      checkpoint_path = tf.train.latest_checkpoint(checkpoint_dir)
      assert checkpoint_path is not None, "Can't find the latest checkpoint under {}".format(checkpoint_dir)

  model = HypergraphModel(num_processing_steps=cfg.NUM_PROCESSING_STEPS,
                          node_output_size=cfg.NODE_OUTPUT_SIZE,
                          force_symmetry=cfg.FORCE_SYMMETRY)

  model.load_weights(checkpoint_path)

  graph_creator = data_utils.get_graph_creator(cfg=cfg)
  predict = _get_predict_function_helper(model, graph_creator, device)

  if compiled:
    problem_generator = data_utils.get_problem_generator(cfg=cfg)
    problem = problem_generator.generate_problem(cfg.SETTINGS_SIMPLE)
    predict(problem)

  return predict
