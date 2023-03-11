import numpy as np
import time

from utils import pascal_voc
from utils.make_matching_problem_pascal_voc import make_matching_problem
from utils.make_matching_problem_pascal_voc import count_common_points
from utils.make_matching_problem_pascal_voc import convert_to_directed

from utils.utils import set_random_seed
from utils.eval_hypergraph_matching import evaluate


def evaluate_one_problem(algorithms,
                         xml_file1,
                         xml_file2,
                         cropped,
                         shuffle_points,
                         shuffle_assignment,
                         scale,
                         rng=None):
  n_algorithm = len(algorithms)

  accuracy_history = np.zeros((n_algorithm,))
  match_score_history = np.zeros((n_algorithm,))
  time_history = np.zeros((n_algorithm,))

  problem = make_matching_problem(xml_file1=xml_file1,
                                  xml_file2=xml_file2,
                                  cropped=cropped,
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


def main(num_pairs_per_category, cropped, shuffle_points, shuffle_assignment, scale):
  print('=' * 50, flush=True)
  print('num_pairs_per_category: {}'.format(num_pairs_per_category))
  print('cropped: {}'.format(cropped))
  print('shuffle_points: {}'.format(shuffle_points))
  print('shuffle_assignment: {}'.format(shuffle_assignment))
  print('scale: {}'.format(scale))
  print('')

  from hypergraph_matching_algorithms import get_algorithms
  algorithms = get_algorithms(dataset_name="Pascal VOC")

  categories = pascal_voc.CATEGORIES

  n_category = len(categories)
  n_algorithm = len(algorithms)

  accuracy_history = np.zeros((n_category, n_algorithm, num_pairs_per_category))
  match_score_history = np.zeros((n_category, n_algorithm, num_pairs_per_category))
  time_history = np.zeros((n_category, n_algorithm, num_pairs_per_category))

  min_num_keypoints = pascal_voc.MIN_NUM_KEYPOINTS
  xml_files_all = pascal_voc.get_all_xml_files(pascal_voc.TEST_LIST_DIR, min_num_keypoints=min_num_keypoints)

  ss = np.random.SeedSequence(12345)
  seeds = ss.spawn(n_category)
  rngs = [np.random.default_rng(s) for s in seeds]

  print('=' * 50, flush=True)
  for cid, (category, rng) in enumerate(zip(categories, rngs)):
    print(category, flush=True)
    xml_files = xml_files_all[category]
    for idx in range(num_pairs_per_category):
      # Randomly choice a pair of samples
      while True:
        idx1, idx2 = rng.choice(len(xml_files), size=2, replace=True)
        xml_file1 = xml_files[idx1]
        xml_file2 = xml_files[idx2]
        n_points = count_common_points(xml_file1, xml_file2)
        if n_points >= min_num_keypoints:
          break

      acc, score, t = evaluate_one_problem(algorithms,
                                           xml_file1=xml_file1,
                                           xml_file2=xml_file2,
                                           cropped=cropped,
                                           shuffle_points=shuffle_points,
                                           shuffle_assignment=shuffle_assignment,
                                           scale=scale,
                                           rng=rng)
      accuracy_history[cid, :, idx] = acc
      match_score_history[cid, :, idx] = score
      time_history[cid, :, idx] = t
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
  main(num_pairs_per_category=1000, cropped=True, shuffle_points=False, shuffle_assignment=True, scale=None)
