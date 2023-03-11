import argparse
import time

import numpy as np
import matplotlib.pyplot as plt

from utils import willow
from utils.make_matching_problem_willow import make_matching_problem
from utils.make_matching_problem_willow import convert_to_directed

from utils.utils import set_random_seed
from utils.eval_hypergraph_matching import evaluate


def evaluate_one_problem(algorithms, mat_file1, mat_file2, n1, n2, shuffle, scale, rng=None):
  n_algorithm = len(algorithms)

  accuracy_history = np.zeros((n_algorithm,))
  match_score_history = np.zeros((n_algorithm,))
  time_history = np.zeros((n_algorithm,))

  problem = make_matching_problem(mat_file1, mat_file2, n1, n2, shuffle, scale, rng=rng)
  problem_directed = convert_to_directed(problem)
  for j in range(n_algorithm):
    the_time = time.time()
    if algorithms[j].get('directed', False):
      input = problem_directed
    else:
      input = problem
    X = algorithms[j]["model"](input)
    time_history[j] = time.time() - the_time
    # TODO: use problem_directed?
    accuracy, match_score = evaluate(problem, X)
    accuracy_history[j] = accuracy
    match_score_history[j] = match_score
  return accuracy_history, match_score_history, time_history


def evaluate_one_category(algorithms, category, n1, n2, shuffle, scale, n_pairs,
                          print_progress=False, rng=None):
  rng = np.random.default_rng(rng)

  mat_files = willow.get_mat_files_test(category=category)

  index_pairs = rng.choice(len(mat_files), size=(n_pairs, 2), replace=True)

  n_algorithm = len(algorithms)
  accuracy_history = np.zeros((n_algorithm, n_pairs))
  match_score_history = np.zeros((n_algorithm, n_pairs))
  time_history = np.zeros((n_algorithm, n_pairs))

  if print_progress:
    print('[{}]'.format(len(index_pairs)), end='', flush=True)
  for idx, (index1, index2) in enumerate(index_pairs):
    mat_file1 = mat_files[index1]
    mat_file2 = mat_files[index2]
    acc, score, t = evaluate_one_problem(algorithms, mat_file1, mat_file2, n1, n2,
                                         shuffle, scale, rng=rng)
    accuracy_history[:, idx] = acc
    match_score_history[:, idx] = score
    time_history[:, idx] = t
    if print_progress:
      print('.', end='', flush=True)

  return accuracy_history, match_score_history, time_history


def main(categories, num_pairs_per_category, num_outliers, shuffle, scale, rngs=None):
  print('Matplotlib backend: {}'.format(plt.get_backend()))

  print('=' * 50, flush=True)
  print('categories: {}'.format(categories))
  print('num_pairs_per_category: {}'.format(num_pairs_per_category))
  print('num_outliers: {}'.format(num_outliers))
  print('shuffle: {}'.format(shuffle))
  print('scale: {}'.format(scale))
  print('')

  if rngs is None:
    rngs = [None for _ in categories]

  from hypergraph_matching_algorithms import get_algorithms
  algorithms = get_algorithms(dataset_name="Willow")

  n_category = len(categories)
  n_algorithm = len(algorithms)

  accuracy_history = np.zeros((n_category, n_algorithm, num_pairs_per_category))
  match_score_history = np.zeros((n_category, n_algorithm, num_pairs_per_category))
  time_history = np.zeros((n_category, n_algorithm, num_pairs_per_category))

  for cid, (category, rng) in enumerate(zip(categories, rngs)):
    print("Test on {}".format(category))
    n1 = 10
    n2 = n1 + num_outliers
    acc, score, t = evaluate_one_category(algorithms, category,
                                          n1, n2, shuffle, scale,
                                          n_pairs=num_pairs_per_category,
                                          print_progress=True,
                                          rng=rng)
    accuracy_history[cid] = acc
    match_score_history[cid] = score
    time_history[cid] = t
    print('', flush=True)
    for aid, algorithm in enumerate(algorithms):
      print(
          '  {}: accuracy={:.2f}%, score={:.4f}, time={} ms'.format(
              algorithm["name"],
              np.mean(accuracy_history[cid, aid]) * 100,
              np.mean(match_score_history[cid, aid]),
              np.mean(time_history[cid, aid]) * 1000
          ),
          flush=True
      )
    print('', flush=True)

  mean_accuracy = np.mean(accuracy_history, axis=2)
  mean_match_score = np.mean(match_score_history, axis=2)
  mean_time = np.mean(time_history, axis=2)

  for aid, algorithm in enumerate(algorithms):
    print('=' * 50, flush=True)
    print(algorithm["name"], flush=True)
    print('  category: accuracy(%), score, time(ms)', flush=True)
    print('  ' + ('-' * 48), flush=True)
    for cid, category in enumerate(categories):
      print(
          '  {}: {:.2f}, {:.4f}, {}'.format(
              category,
              mean_accuracy[cid, aid] * 100,
              mean_match_score[cid, aid],
              mean_time[cid, aid] * 1000
          ),
          flush=True
      )
    print('  ' + ('-' * 48), flush=True)
    print(
        '  mean: {:.2f}, {:.4f}, {}'.format(
            np.mean(mean_accuracy[:, aid]) * 100,
            np.mean(mean_match_score[:, aid]),
            np.mean(mean_time[:, aid]) * 1000
        ),
        flush=True
    )
    print('', flush=True)

  print('=' * 50, flush=True)
  print('Comparison of matching accuracy (%) on the Willow Object dataset')
  print('-' * 50, flush=True)

  # Draw table head
  print('|Algorithm|', end='', flush=True)
  for category in categories:
    print('{}|'.format(category), end='', flush=True)
  print('AVG|', flush=True)

  # Draw head line
  print('|---------|', end='', flush=True)
  for category in categories:
    print('{}:|'.format('-' * (len(category) - 1)), end='', flush=True)
  print('--:|', flush=True)

  for aid, algorithm in enumerate(algorithms):
    print('|{}|'.format(algorithm["name"]), end='', flush=True)
    for cid, category in enumerate(categories):
      print('{:.1f}|'.format(mean_accuracy[cid, aid] * 100), end='', flush=True)
    print('{:.1f}|'.format(np.mean(mean_accuracy[:, aid]) * 100), flush=True)


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Evaluation on Willow Object Dataset')
  parser.add_argument('--seed', default=12345, type=int, help='Seed for random generator')
  parser.add_argument('--global-seed', default=1024, type=int, help='Seed for the global random generator')
  parser.add_argument('--pairs', default=1000, type=int, help='number of pairs per categories to evaluate')
  parser.add_argument('--outlier', default=0, type=int, help='number of outliers')
  parser.add_argument('--no-shuffle', action='store_true', help='Do not shuffle the order of points')
  parser.add_argument('--scale', default=None, type=float, help='value to compute the affinity')
  parser.add_argument('categories', default=willow.CATEGORIES, type=str, nargs='*', metavar='categories',
                      help='list of categories')
  args = parser.parse_args()

  print('Arguments: ', args)

  set_random_seed(seed=args.global_seed)

  ss = np.random.SeedSequence(args.seed)
  seeds = ss.spawn(len(willow.CATEGORIES))
  rngs_all = [np.random.default_rng(s) for s in seeds]

  rngs = []
  for category in args.categories:
    cid = willow.CATEGORIES.index(category)
    rngs.append(rngs_all[cid])

  main(categories=args.categories,
       num_pairs_per_category=args.pairs,
       num_outliers=args.outlier,
       shuffle=not args.no_shuffle,
       scale=args.scale,
       rngs=rngs)
