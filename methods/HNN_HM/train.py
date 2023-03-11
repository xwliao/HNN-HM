import os
import sys
import time
import random

from typing import List

import numpy as np
import tensorflow as tf

from utils.hungarian import hungarian

from hnn_hm.hypergraph import HypergraphsTuple
from hnn_hm.models import HypergraphModel

from hnn_hm.config.utils import get_config
from hnn_hm.data.data import get_dataloader
from hnn_hm.data.utils import get_dataset


model = None
optimizer = None


def set_gpu_memory_growth():
  gpus = tf.config.experimental.list_physical_devices('GPU')
  for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


def set_random_seed(seed):
  random.seed(seed)
  np.random.seed(seed)
  tf.random.set_seed(seed)


def get_optimizer(cfg):
  assert cfg.OPTIMIZER in ['adam', 'sgd']

  if cfg.OPTIMIZER == 'adam':
    learning_rate = cfg.LEARNING_RATE
    optimizer = tf.keras.optimizers.Adam(learning_rate, amsgrad=cfg.AMSGRAD)
  elif cfg.OPTIMIZER == 'sgd':
    learning_rate = cfg.LEARNING_RATE
    momentum = cfg.MOMENTUM
    optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=momentum)

  return optimizer


def compute_losses(target: HypergraphsTuple, outputs: List[HypergraphsTuple]) -> tf.Tensor:
  bce = tf.keras.losses.BinaryCrossentropy()
  N = len(outputs)
  losses = tf.TensorArray(tf.float32, size=N)
  for i, output in enumerate(outputs):
    loss = bce(target.nodes, output.nodes)
    losses = losses.write(i, loss)
  return losses.stack()


# NOTE: output has None element, so we cannot use tf.function with signature directly!
def compute_loss(target: HypergraphsTuple, outputs: List[HypergraphsTuple]) -> tf.Tensor:
  losses = compute_losses(target, outputs)
  loss = tf.reduce_mean(losses)
  return loss


def compute_accuracy(target: HypergraphsTuple, output: HypergraphsTuple) -> tf.Tensor:
  N = tf.size(target.n_node)
  accuracy_list = tf.TensorArray(tf.float32, size=N)
  start = 0
  target_shape = tf.stack([target.n_row, target.n_col], axis=-1)
  for i in tf.range(N):
    n = target.n_node[i]
    shape = target_shape[i]

    P = tf.reshape(output.nodes[start:start + n], shape)
    P = tf.numpy_function(hungarian, [P], P.dtype)
    P = tf.cast(P, tf.float32)

    T = tf.reshape(target.nodes[start:start + n], shape)
    T = tf.cast(T, tf.float32)

    pt = tf.reduce_sum(tf.multiply(P, T))
    tt = tf.reduce_sum(tf.multiply(T, T))
    acc = pt / tt
    accuracy_list = accuracy_list.write(i, acc)

    start += n

  accuries = accuracy_list.stack()
  out = tf.reduce_mean(accuries)
  return out


# @tf.function(input_signature=[cfg.INPUT_SIGNATURE, cfg.TARGET_SIGNATURE])
def train_step(inputs, targets):
  global model, optimizer

  with tf.GradientTape() as tape:
    # A list of outputs, one per processing step.
    outputs = model(inputs)

    # Loss.
    loss = compute_loss(targets, outputs)

  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))

  return loss


# @tf.function(input_signature=[cfg.INPUT_SIGNATURE, cfg.TARGET_SIGNATURE])
def evaluate_batch(inputs, targets):
  global model
  n_samples = tf.size(targets.n_node)
  outputs = model(inputs)
  loss = compute_loss(targets, outputs)
  # Evaluate only the last output
  accuracy = compute_accuracy(targets, outputs[-1])
  return loss, accuracy, n_samples


# @tf.function
def evaluate_dataset(dataset: tf.data.Dataset):
  n_samples = tf.constant(0, tf.int32)
  loss_sum = tf.constant(0, tf.float32)
  accuracy_sum = tf.constant(0, tf.float32)

  for inputs, targets in dataset:
    loss, accuracy, n = evaluate_batch(inputs, targets)
    n_samples += n
    n = tf.cast(n, tf.float32)
    loss_sum += loss * n
    accuracy_sum += accuracy * n

  N = tf.cast(n_samples, tf.float32)
  loss = loss_sum / N
  accuracy = accuracy_sum / N

  return loss, accuracy, n_samples


def main(cfg):
  global model, optimizer
  global train_step, evaluate_batch

  set_gpu_memory_growth()
  set_random_seed(cfg.GLOBAL_RANDOM_SEED)

  model = HypergraphModel(num_processing_steps=cfg.NUM_PROCESSING_STEPS,
                          node_output_size=cfg.NODE_OUTPUT_SIZE,
                          force_symmetry=cfg.FORCE_SYMMETRY)

  optimizer = get_optimizer(cfg)

  log_dir = cfg.LOG_DIR
  writer = tf.summary.create_file_writer(log_dir)

  checkpoint_dir = cfg.CHECKPOINT_DIR
  os.makedirs(checkpoint_dir, exist_ok=True)
  checkpoint_path_template = os.path.join(checkpoint_dir, "cp-{}.ckpt")

  start_time = time.time()
  last_log_time = start_time

  _dataset_tr = get_dataset(sets="train", cfg=cfg, rng=cfg.RNG_TR)
  dataset_tr = get_dataloader(_dataset_tr,
                              cfg.NUM_ITERATIONS_TR,
                              cfg.BATCH_SIZE_TR,
                              cfg.SETTINGS_TR,
                              num_parallel_calls=cfg.NUM_PARALLEL_CALLS,
                              deterministic=cfg.DETERMINISTIC_TR)
  if getattr(cfg, 'DATASET_CACHE_TR', None) is not None:
      dataset_tr = dataset_tr.cache(cfg.DATASET_CACHE_TR)
  dataset_tr = dataset_tr.prefetch(cfg.PREFETCH_TR)

  _dataset_ge = get_dataset(sets="val", cfg=cfg, rng=cfg.RNG_GE)
  dataset_ge = get_dataloader(_dataset_ge,
                              cfg.NUM_ITERATIONS_GE,
                              cfg.BATCH_SIZE_GE,
                              cfg.SETTINGS_GE,
                              num_parallel_calls=cfg.NUM_PARALLEL_CALLS,
                              deterministic=cfg.DETERMINISTIC_GE)
  if getattr(cfg, 'DATASET_CACHE_GE', None) is not None:
      dataset_ge = dataset_ge.cache(cfg.DATASET_CACHE_GE)
  dataset_ge = dataset_ge.prefetch(cfg.PREFETCH_GE)

  print("# (iteration number), "
        "T (elapsed seconds), "
        "L (training loss before update), "
        "Ltr (training loss), "
        "Atr (training accuracy), "
        "Lge (test/generalization loss), "
        "Age (test/generalization accuracy), "
        "LR (current learning rate)",
        flush=True)

  train_step = tf.function(train_step,
                           input_signature=[cfg.INPUT_SIGNATURE, cfg.TARGET_SIGNATURE])
  evaluate_batch = tf.function(evaluate_batch,
                               input_signature=[cfg.INPUT_SIGNATURE, cfg.TARGET_SIGNATURE])
  # train_step = (tf.function(train_step)
  #               .get_concrete_function(cfg.INPUT_SIGNATURE, cfg.TARGET_SIGNATURE))
  # evaluate_batch = (tf.function(evaluate_batch)
  #                   .get_concrete_function(cfg.INPUT_SIGNATURE, cfg.TARGET_SIGNATURE))

  iteration = 0
  for inputs_tr, targets_tr in dataset_tr:
    iteration = iteration + 1

    lr = optimizer._decayed_lr(tf.float32)

    loss = train_step(inputs_tr, targets_tr)

    the_time = time.time()

    with writer.as_default():
      tf.summary.scalar('lr', lr, step=iteration - 1)
      tf.summary.scalar('loss', loss, step=iteration - 1)

    should_log = (iteration == cfg.NUM_ITERATIONS_TR)
    if not should_log and getattr(cfg, 'LOG_EVERY_SECONDS', None) is not None:
        elapsed_since_last_log = the_time - last_log_time
        should_log = (elapsed_since_last_log > cfg.LOG_EVERY_SECONDS)
    if not should_log and getattr(cfg, 'LOG_EVERY_ITERATIONS', None) is not None:
        should_log = (iteration % cfg.LOG_EVERY_ITERATIONS == 0)

    if should_log:
      last_log_time = the_time

      # Evaluate on the training batch
      # Note:
      # The following loss is different from the loss computed by train_step.
      # Because loss computed by train_step is based on old parameters,
      # while the loss computed following is based on updated parameters.
      loss_tr, accuracy_tr, _ = (
          x.numpy() for x in evaluate_batch(inputs_tr, targets_tr)
      )

      # Test/generalization
      loss_ge, accuracy_ge, _ = (
          x.numpy() for x in evaluate_dataset(dataset_ge)
      )

      elapsed = time.time() - start_time

      lr = lr.numpy()

      print("# {:07d}, T {:.1f}, L {:.4f}, "
            "Ltr {:.4f}, Atr {:.4f}, "
            "Lge {:.4f}, Age {:.4f}, "
            "LR {:.6f}".format(
                iteration, elapsed, loss,
                loss_tr, accuracy_tr,
                loss_ge, accuracy_ge,
                lr
            ),
            flush=True)

      with writer.as_default():
        tf.summary.scalar('T', elapsed, step=iteration,
                          description='elapsed seconds')

        # Train
        tf.summary.scalar('Ltr', loss_tr, step=iteration,
                          description='training loss')
        tf.summary.scalar('Atr', accuracy_tr, step=iteration,
                          description='training accuracy')

        # Validation
        tf.summary.scalar('Lge', loss_ge, step=iteration,
                          description='test/generalization loss')
        tf.summary.scalar('Age', accuracy_ge, step=iteration,
                          description='test/generalization accuracy')
      writer.flush()

      model.save_weights(checkpoint_path_template.format(iteration))


if __name__ == '__main__':
  assert len(sys.argv) > 1, f"Usage: {sys.argv[0]} <dataset_name>"
  dataset_name = sys.argv[1]
  cfg = get_config(dataset_name=dataset_name)
  main(cfg)
