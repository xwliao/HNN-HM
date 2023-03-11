import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

from utils import willow
from utils.make_matching_problem_willow import make_matching_problem
from utils.make_matching_problem_willow import convert_to_directed

from utils.hungarian import hungarian
from utils.eval_hypergraph_matching import evaluate

from utils.visualization import plot_results
from utils.visualization import Hypergraph

from utils.utils import set_random_seed


def create_graph(points):
  nodes = points
  edges = None
  hyperedges = None
  n_node = points.shape[0]
  n_edge = 0
  graph = Hypergraph(nodes=nodes, edges=edges,
                     hyperedges=hyperedges,
                     n_node=n_node, n_edge=n_edge)
  return graph


def show(ax, points1, points2, image1, image2, prediction, target, image_pad_value=0, title=None):
  def pad(arr, pad_h, value):
    h = arr.shape[0]
    shape = list(arr.shape)   # (H, W, C) or (H, W)
    shape[0] = h + pad_h  # pad value height
    out = np.full(shape, value, dtype=arr.dtype)
    out[:h] = arr
    return out

  h1 = image1.shape[0]
  h2 = image2.shape[0]

  if h1 < h2:
    image1 = pad(image1, h2 - h1, value=image_pad_value)
  elif h1 > h2:
    image2 = pad(image2, h1 - h2, value=image_pad_value)

  image = np.concatenate([image1, image2], axis=1)
  ax.axis('off')
  ax.imshow(image)

  graph1 = create_graph(points1)
  graph2 = create_graph(points2)
  # (x_offset, y_offset)
  graph1_offset = (0, 0)
  graph2_offset = (image1.shape[1], 0)
  plot_results(ax, graph1, graph2,
               assignmatrix=prediction,
               target=target,
               title=title,
               graph1_offset=graph1_offset,
               graph2_offset=graph2_offset)


def evaluate_one_problem(algorithms, mat_file1, mat_file2, n1, n2, shuffle, scale,
                         save_dir=None, figsize=(12.8, 9.6), rng=None):
  problem = make_matching_problem(
      mat_file1=mat_file1, mat_file2=mat_file2,
      n1=n1, n2=n2,
      shuffle=shuffle, scale=scale,
      rng=rng
  )
  problem_directed = convert_to_directed(problem)

  target = np.array(problem["assignmentMatrix"], dtype=np.int32)

  points1 = problem["P1"]
  points2 = problem["P2"]

  image1 = willow.get_image(mat_file1)
  image2 = willow.get_image(mat_file2)

  if save_dir is not None:
    save_dir = Path(save_dir)
    if not save_dir.exists():
      save_dir.mkdir(parents=True)

  for algorithm in algorithms:
    if algorithm.get('directed', False):
      input = problem_directed
    else:
      input = problem
    X = algorithm["model"](input)

    X = hungarian(X)

    # TODO: use problem_directed?
    accuracy, match_score = evaluate(problem, X, lap_solver='identity')

    fig, ax = plt.subplots(figsize=figsize)

    title = '{}, accuracy: {:.2f}%, matching score: {:.4f}'.format(
        algorithm["name"], accuracy * 100, match_score
    )

    show(ax, points1, points2, image1, image2, X, target, image_pad_value=0, title=title)

    if save_dir is not None:
      fname = 'alg_{}_willow_file1_{}_file2_{}_n1_{}_n2_{}_shuffle_{}_scale_{}_acc_{}_score_{}.pdf'.format(
          algorithm["name"],
          Path(mat_file1).name,
          Path(mat_file2).name,
          n1, n2,
          shuffle, scale, accuracy * 100, match_score
      )
      fig.savefig(save_dir / fname, bbox_inches='tight')

    if save_dir is None:
      plt.show()


def main(num_pairs_per_category):
  from hypergraph_matching_algorithms import get_algorithms
  algorithms = get_algorithms(dataset_name="Willow")

  print('Matplotlib backend: {}'.format(plt.get_backend()))

  categories = willow.CATEGORIES

  # Spawn a random generator for each category
  ss = np.random.SeedSequence(12345)
  seeds = ss.spawn(len(categories))
  rngs = [np.random.default_rng(s) for s in seeds]

  mat_files_all = {}
  index_pairs_all = {}
  for category, rng in zip(categories, rngs):
    mat_files = willow.get_mat_files_test(category)
    index_pairs = rng.choice(len(mat_files), size=(num_pairs_per_category, 2), replace=True)

    mat_files_all[category] = mat_files
    index_pairs_all[category] = index_pairs

  # The scale value is related to the computation of the hyperedge affinity.
  # Test the same pair of samples with different scales and #outliers.
  for scale in [None, 0.2, 0.5, 1.0]:
    for n_outlier in [0, 5, 10, 15, 20]:
      # For each category, spawn the same random generator for different scales and different #outliers(?)
      child_seeds = [s.spawn(1)[0] for s in seeds]
      child_rngs = [np.random.default_rng(s) for s in child_seeds]
      for category, rng in zip(categories, child_rngs):
        mat_files = mat_files_all[category]
        index_pairs = index_pairs_all[category]
        for idx, (idx1, idx2) in enumerate(index_pairs):
          mat_file1 = mat_files[idx1]
          mat_file2 = mat_files[idx2]
          n1 = 10
          n2 = 10 + n_outlier
          shuffle = True
          save_dir = Path('..') / 'results' / 'demo_willow' / 'scale_{}'.format(scale) / 'out_{}'.format(n_outlier) / category / 'pair{}'.format(idx)  # noqa
          # save_dir = None  # TODO:
          evaluate_one_problem(algorithms,
                               mat_file1=mat_file1, mat_file2=mat_file2,
                               n1=n1, n2=n2,
                               shuffle=shuffle, scale=scale,
                               save_dir=save_dir,
                               rng=rng)


if __name__ == '__main__':
  set_random_seed(seed=1024)
  main(num_pairs_per_category=10)
