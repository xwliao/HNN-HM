import time
from pathlib import Path

import numpy as np

from utils import house
from utils.make_matching_problem_house import make_matching_problem
from utils.make_matching_problem_house import convert_to_directed

from utils.eval_hypergraph_matching import evaluate

from evaluation_utils import show
from evaluation_utils import create_directory
from evaluation_utils import read_pickle
from evaluation_utils import write_pickle


def evaluate_one_problem(algorithms, index1, index2, nP1, nP2, shuffle_points, shuffle_assignment, scale, rng=None):
  n_algorithm = len(algorithms)

  accuracy_history = np.zeros((n_algorithm,))
  match_score_history = np.zeros((n_algorithm,))
  time_history = np.zeros((n_algorithm,))

  problem = make_matching_problem(index1, index2, nP1, nP2,
                                  shuffle_points=shuffle_points,
                                  shuffle_assignment=shuffle_assignment,
                                  scale=scale,
                                  rng=rng)
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


def evaluate_one_gap(algorithms, gap, nP1, nP2, shuffle_points, shuffle_assignment, scale,
                     print_progress=False, rng=None):
  index_pairs = []
  for index1 in range(house.DATASET_SIZE - gap):
    index2 = index1 + gap
    index_pairs.append((index1, index2))

  accuracy_history = []
  match_score_history = []
  time_history = []

  if print_progress:
    print('[{}]'.format(len(index_pairs)), end='', flush=True)
  for (index1, index2) in index_pairs:
    acc, score, t = evaluate_one_problem(algorithms, index1, index2, nP1, nP2,
                                         shuffle_points=shuffle_points,
                                         shuffle_assignment=shuffle_assignment,
                                         scale=scale,
                                         rng=rng)
    accuracy_history.append(acc)
    match_score_history.append(score)
    time_history.append(t)
    if print_progress:
      print('.', end='', flush=True)

  return accuracy_history, match_score_history, time_history


def compute(algorithms, value_list, settings, key, rng=None):
  accuracy_history = []
  match_score_history = []
  time_history = []

  cnt = 0
  tot = len(value_list)
  for value in value_list:
    cnt += 1
    print('{}/{}: '.format(cnt, tot), end='', flush=True)
    if key == 'gap':
      gap = value
      nP1 = settings["nP1"]
      nP2 = settings["nP2"]
      shuffle_points = settings["shuffle_points"]
      shuffle_assignment = settings["shuffle_assignment"]
      scale = settings["scale"]
    elif key == 'nOutlier':
      gap = settings["gap"]
      nP2 = settings["nP2"]
      shuffle_points = settings["shuffle_points"]
      shuffle_assignment = settings["shuffle_assignment"]
      nP1 = nP2 - value
      scale = settings["scale"]
    acc, score, t = evaluate_one_gap(algorithms, gap, nP1, nP2,
                                     shuffle_points=shuffle_points,
                                     shuffle_assignment=shuffle_assignment,
                                     scale=scale, print_progress=True,
                                     rng=rng)
    accuracy_history.append(np.mean(acc, axis=0))
    match_score_history.append(np.mean(score, axis=0))
    time_history.append(np.mean(t, axis=0))
    print('', flush=True)

  # Plot Results
  mean_accuracy = np.array(accuracy_history)
  mean_match_score = np.array(match_score_history)
  mean_time = np.array(time_history)

  return mean_accuracy, mean_match_score, mean_time


def show_pickle(pickle_path, **kwargs):
  res = read_pickle(pickle_path)
  res.update(kwargs)
  show(**res)


def run(settings, key, xlabel=None, text_list=None, save_dir=None, rng=None):
  from hypergraph_matching_algorithms import get_algorithms
  algorithms = get_algorithms(dataset_name="House")

  value_list = settings[key]

  mean_accuracy, mean_match_score, mean_time = compute(algorithms=algorithms,
                                                       value_list=value_list,
                                                       settings=settings,
                                                       key=key,
                                                       rng=rng)

  for algorithm in algorithms:
    del algorithm["model"]

  res = dict(algorithms=algorithms,
             xdata=value_list,
             mean_accuracy=mean_accuracy,
             mean_match_score=mean_match_score,
             mean_time=mean_time,
             xlabel=xlabel,
             text_list=text_list)

  if save_dir is not None:
    create_directory(save_dir)
    pickle_path = Path(save_dir) / 'results.pickle'
    write_pickle(pickle_path, res)
    show_pickle(pickle_path, save_dir=save_dir)
  else:
    show(**res)
