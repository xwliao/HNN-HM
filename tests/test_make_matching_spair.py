import random
import numpy as np
from utils import spair
from utils.make_matching_problem_spair import make_matching_problem
from utils.make_matching_problem_spair import convert_to_directed


def set_random_seed(seed):
  random.seed(seed)
  np.random.seed(seed)


def assert_same_problems(problem1, problem2):
  def assert_allclose(v1, v2, *args, **kwargs):
    if (v1 is None) or (v2 is None):
      assert v1 is None
      assert v2 is None
    else:
      np.testing.assert_allclose(v1, v2, *args, **kwargs)
  assert problem1["nP1"] == problem2["nP1"]
  assert problem1["nP2"] == problem2["nP2"]
  assert_allclose(problem1["P1"], problem2["P1"], rtol=1e-4, atol=0)
  assert_allclose(problem1["P2"], problem2["P2"], rtol=1e-4, atol=0)
  assert_allclose(problem1["indH1"], problem2["indH1"], rtol=1e-4, atol=0)
  assert_allclose(problem1["valH1"], problem2["valH1"], rtol=1e-4, atol=0)
  assert_allclose(problem1["indH2"], problem2["indH2"], rtol=1e-4, atol=0)
  assert_allclose(problem1["valH2"], problem2["valH2"], rtol=1e-4, atol=0)
  assert_allclose(problem1["indH3"], problem2["indH3"], rtol=1e-4, atol=0)
  assert_allclose(problem1["valH3"], problem2["valH3"], rtol=1e-4, atol=0)
  assert_allclose(problem1["assignmentMatrix"], problem2["assignmentMatrix"], rtol=1e-4, atol=0)


def test_make_matching_problem():
  set_random_seed(seed=1024)

  dataset = spair.SPair71k(spair.get_dataset("train"))
  dataset = spair.filter_points(dataset, min_num_keypoints=3)

  problem = make_matching_problem(
      data=next(dataset),
      scale=0.2
  )
  print(problem)
  problem_directed = convert_to_directed(problem)
  print(problem_directed)


def _test_make_matching_problem_random_helper(sets, num_samples, cls=None):
  set_random_seed(seed=1024)

  seed = 12345
  min_num_keypoints = 3

  rngs = []
  datasets = []
  for _ in range(2):
    rng = np.random.default_rng(seed)
    dataset = spair.SPair71k(spair.get_dataset(sets, rng=rng), cls=cls)
    dataset = spair.filter_points(dataset, min_num_keypoints=min_num_keypoints)
    rngs.append(rng)
    datasets.append(dataset)

  for idx in range(num_samples):
    problems = []
    for dataset, rng in zip(datasets, rngs):
      problem = make_matching_problem(data=next(dataset), scale=0.2, rng=rng)
      problems.append(problem)
    for problem in problems[1:]:
      assert_same_problems(problems[0], problem)


def test_make_matching_problem_random():
  _test_make_matching_problem_random_helper("train", 100)
  _test_make_matching_problem_random_helper("val", 100)
  _test_make_matching_problem_random_helper("test", 100, cls='aeroplane')


if __name__ == '__main__':
  test_make_matching_problem()
  test_make_matching_problem_random()
