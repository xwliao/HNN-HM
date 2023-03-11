import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path

from utils import house
from utils.make_matching_problem_house import make_matching_problem
from utils.make_matching_problem_house import convert_to_directed

from utils.hungarian import hungarian
from utils.eval_hypergraph_matching import evaluate

from utils.visualization import plot_results
from utils.visualization import Hypergraph


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
  ax.imshow(image, cmap='gray')

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


def evaluate_one_problem(algorithms, index1, index2, nP1, nP2,
                         shuffle_points, shuffle_assignment,
                         scale, save_dir=None, figsize=(12.8, 9.6),
                         rng=None):

  problem = make_matching_problem(index1, index2, nP1, nP2,
                                  shuffle_points=shuffle_points,
                                  shuffle_assignment=shuffle_assignment,
                                  scale=scale,
                                  rng=rng)
  problem_directed = convert_to_directed(problem)

  target = np.array(problem["assignmentMatrix"], dtype=np.int32)

  points1 = problem["P1"]
  points2 = problem["P2"]

  image1 = house.get_image(index1)
  image2 = house.get_image(index2)

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
      fname = ('alg_{}_house'
               '_ind1_{}_ind2_{}'
               '_n1_{}_n2_{}'
               '_shuffle_points_{}_shuffle_assignment_{}'
               '_scale_{}'
               '_acc_{}_score_{}.pdf').format(
          algorithm["name"], index1, index2, nP1, nP2,
          shuffle_points, shuffle_assignment, scale, accuracy * 100, match_score
      )
      fig.savefig(save_dir / fname, bbox_inches='tight')

    if save_dir is None:
      plt.show()


def main():
  from hypergraph_matching_algorithms import get_algorithms
  algorithms = get_algorithms(dataset_name="House")

  # save_dir = Path('..') / 'results' / 'demo_house' / 'gap_1'
  # evaluate_one_problem(algorithms, index1=0, index2=1, nP1=4, nP2=4,
  #                      shuffle_points=False, shuffle_assignment=True,
  #                      scale=None, save_dir=save_dir, rng=rng)

  for scale in [None, 0.2, 0.5, 1.0]:
    rng = np.random.default_rng(12345)
    for gap in [1, 20, 50, 80, 100]:
      for nOutlier in [0, 10, 20]:
        index1 = 0
        index2 = index1 + gap
        nP2 = 30
        nP1 = nP2 - nOutlier
        shuffle_points = False
        shuffle_assignment = True
        save_dir = Path('..') / 'results' / 'demo_house' / '{}'.format(scale) / 'gap_{}_out_{}'.format(gap, nOutlier)
        evaluate_one_problem(algorithms,
                             index1=index1, index2=index2,
                             nP1=nP1, nP2=nP2,
                             shuffle_points=shuffle_points,
                             shuffle_assignment=shuffle_assignment,
                             scale=scale,
                             save_dir=save_dir,
                             rng=rng)


if __name__ == '__main__':
  main()
