import time
import numpy as np

from utils import spair
from utils.make_matching_problem_spair import make_matching_problem
from utils.make_matching_problem_spair import convert_to_directed

from utils.utils import set_random_seed
from utils.eval_hypergraph_matching import evaluate


def evaluate_one_problem(algorithms, data, scale, rng=None):
  n_algorithm = len(algorithms)

  accuracy_history = np.zeros((n_algorithm,))
  match_score_history = np.zeros((n_algorithm,))
  time_history = np.zeros((n_algorithm,))

  problem = make_matching_problem(data=data, scale=scale, rng=rng)
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


def main(scale):
  print('=' * 50, flush=True)
  print('scale: {}'.format(scale))
  print('')

  from hypergraph_matching_algorithms import get_algorithms
  algorithms = get_algorithms(dataset_name="SPair-71k")

  rng = np.random.default_rng(12345)
  dataset_all = spair.get_dataset("test", rng=rng)
  categories = dataset_all.classes

  n_category = len(categories)
  n_algorithm = len(algorithms)

  number_history = np.zeros((n_category, n_algorithm))
  accuracy_history = np.zeros((n_category, n_algorithm))
  match_score_history = np.zeros((n_category, n_algorithm))
  time_history = np.zeros((n_category, n_algorithm))

  min_num_keypoints = 3

  print('=' * 50, flush=True)
  for cid, category in enumerate(categories):
    print(category, flush=True)
    dataset = spair.SPair71k(dataset_all, cls=category, shuffle=True)
    dataset = spair.filter_points(dataset, min_num_keypoints=min_num_keypoints)
    for data in dataset:
      acc, score, t = evaluate_one_problem(algorithms, data, scale, rng=rng)

      number_history[cid, :] += 1
      accuracy_history[cid, :] += acc
      match_score_history[cid, :] += score
      time_history[cid, :] += t

    for aid, algorithm in enumerate(algorithms):
      print(
          '  {}: accuracy={:.2f}%, score={:.4f}, time={} ms'.format(
              algorithm["name"],
              (accuracy_history[cid, aid] / number_history[cid, aid]) * 100,
              (match_score_history[cid, aid] / number_history[cid, aid]),
              (time_history[cid, aid] / number_history[cid, aid]) * 1000
          ),
          flush=True
      )
    print('', flush=True)

  mean_accuracy = accuracy_history / number_history
  mean_match_score = match_score_history / number_history
  mean_time = time_history / number_history

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
  print('Comparison of matching accuracy (%) on the Pascal VOC dataset')
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
  set_random_seed(seed=1234)
  main(scale=None)
