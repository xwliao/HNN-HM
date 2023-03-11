import re
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from utils import pascal_voc
from utils.make_matching_problem_pascal_voc import make_matching_problem
from utils.make_matching_problem_pascal_voc import count_common_points
from utils.make_matching_problem_pascal_voc import convert_to_directed

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


def show(ax, points1, points2, image1, image2, prediction, target, title=None):
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


def evaluate_one_problem(algorithms, xml_file1, xml_file2, cropped,
                         shuffle_points, shuffle_assignment, scale,
                         save_dir=None, figsize=(12.8, 9.6),
                         rng=None):
  problem = make_matching_problem(xml_file1=xml_file1,
                                  xml_file2=xml_file2,
                                  cropped=cropped,
                                  shuffle_points=shuffle_points,
                                  shuffle_assignment=shuffle_assignment,
                                  scale=scale,
                                  rng=rng)
  problem_directed = convert_to_directed(problem)

  target = np.array(problem["assignmentMatrix"], dtype=np.int32)

  points1 = problem["P1"]
  points2 = problem["P2"]

  image1 = pascal_voc.get_image(xml_file1, cropped=cropped)
  image2 = pascal_voc.get_image(xml_file2, cropped=cropped)

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

    show(ax, points1, points2, image1, image2, X, target, title=title)

    if save_dir is not None:
      fname = ('alg_{}_pascal_voc'
               '_file1_{}_file2_{}'
               '_cropped_{}'
               '_shuffle_points_{}'
               '_shuffle_assignment_{}'
               '_scale_{}'
               '_acc_{}_score_{}.pdf').format(
          algorithm["name"],
          re.sub(r'[\\\/]', r'_', xml_file1),
          re.sub(r'[\\\/]', r'_', xml_file2),
          cropped,
          shuffle_points,
          shuffle_assignment,
          scale,
          accuracy * 100,
          match_score
      )
      fig.savefig(save_dir / fname, bbox_inches='tight')

    if save_dir is None:
      plt.show()


def main(num_pairs_per_category):
  from hypergraph_matching_algorithms import get_algorithms
  algorithms = get_algorithms(dataset_name="Pascal VOC")

  min_num_keypoints = pascal_voc.MIN_NUM_KEYPOINTS
  xml_files_all = pascal_voc.get_all_xml_files(pascal_voc.TEST_LIST_DIR, min_num_keypoints=min_num_keypoints)

  categories = pascal_voc.CATEGORIES

  # Spawn a random generator for each category
  ss = np.random.SeedSequence(12345)
  seeds = ss.spawn(len(categories))
  rngs = [np.random.default_rng(s) for s in seeds]

  index_pairs_all = {}
  for category, rng in zip(categories, rngs):
    xml_files = xml_files_all[category]
    index_pairs_all[category] = []
    for _ in range(num_pairs_per_category):
      # Randomly choice a pair of samples
      while True:
        idx1, idx2 = rng.choice(len(xml_files), size=2, replace=True)
        xml_file1 = xml_files[idx1]
        xml_file2 = xml_files[idx2]
        n_points = count_common_points(xml_file1, xml_file2)
        if n_points >= min_num_keypoints:
          break
      index_pairs_all[category].append((idx1, idx2))

  # The scale value is related to the computation of the hyperedge affinity.
  # Test the same pair of samples with different scales.
  for scale in [None, 0.2, 0.5, 1.0]:
    # For each category, spawn the same random generator for different scales
    child_seeds = [s.spawn(1)[0] for s in seeds]
    child_rngs = [np.random.default_rng(s) for s in child_seeds]

    for category, rng in zip(categories, child_rngs):
      xml_files = xml_files_all[category]
      index_pairs = index_pairs_all[category]
      for idx, (idx1, idx2) in enumerate(index_pairs):
        xml_file1 = xml_files[idx1]
        xml_file2 = xml_files[idx2]
        cropped = True
        shuffle_points = False
        shuffle_assignment = True
        save_dir = Path('..') / 'results' / 'demo_pascal_voc' / '{}'.format(scale) / category / 'pair{}'.format(idx)
        # save_dir = None
        evaluate_one_problem(algorithms,
                             xml_file1=xml_file1,
                             xml_file2=xml_file2,
                             cropped=cropped,
                             shuffle_points=shuffle_points,
                             shuffle_assignment=shuffle_assignment,
                             scale=scale,
                             save_dir=save_dir,
                             rng=rng)


if __name__ == '__main__':
  set_random_seed(seed=1234)
  main(num_pairs_per_category=10)
