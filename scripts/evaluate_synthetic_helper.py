import time
from pathlib import Path

import numpy as np

from utils.make_matching_problem_synthetic import make_matching_problem
from utils.make_matching_problem_synthetic import convert_to_directed

from utils.eval_hypergraph_matching import evaluate

from evaluation_utils import show
from evaluation_utils import create_directory
from evaluation_utils import read_pickle
from evaluation_utils import write_pickle


def compute(algorithms, n_test, value_list, settings_template, key, rng=None):
  n_setting = len(value_list)
  n_algorithm = len(algorithms)

  accuracy_history = np.zeros((n_setting, n_algorithm, n_test))
  match_score_history = np.zeros((n_setting, n_algorithm, n_test))
  time_history = np.zeros((n_setting, n_algorithm, n_test))

  for k in range(n_test):
    print('Test: {} of {} '.format(k + 1, n_test), end='', flush=True)
    for i, value in enumerate(value_list):
      settings = settings_template.copy()
      settings[key] = value
      problem = make_matching_problem(settings, rng=rng)
      problem_directed = convert_to_directed(problem)
      for j in range(n_algorithm):
        the_time = time.time()
        if algorithms[j].get('directed', False):
          input = problem_directed
        else:
          input = problem
        X = algorithms[j]["model"](input)
        time_history[i, j, k] = time.time() - the_time
        # TODO: use problem_directed?
        accuracy, match_score = evaluate(problem, X)
        accuracy_history[i, j, k] = accuracy
        match_score_history[i, j, k] = match_score
      print('.', end='', flush=True)
    print('', flush=True)

  # Plot Results
  mean_accuracy = np.mean(accuracy_history, axis=2)
  mean_match_score = np.mean(match_score_history, axis=2)
  mean_time = np.mean(time_history, axis=2)

  return mean_accuracy, mean_match_score, mean_time


def show_pickle(pickle_path, **kwargs):
  res = read_pickle(pickle_path)
  res.update(kwargs)
  show(**res)


def run(n_test, settings_template, key, xlabel=None, text_list=None, save_dir=None, rng=None):
  from hypergraph_matching_algorithms import get_algorithms
  algorithms = get_algorithms(dataset_name="Synthetic")

  value_list = settings_template[key]

  mean_accuracy, mean_match_score, mean_time = compute(algorithms=algorithms,
                                                       n_test=n_test,
                                                       value_list=value_list,
                                                       settings_template=settings_template,
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
