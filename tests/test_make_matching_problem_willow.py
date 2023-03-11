import os
import random
import numpy as np

from utils import willow
from utils.make_matching_problem_willow import make_matching_problem
from utils.make_matching_problem_willow import convert_to_directed


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

  mat_file1 = os.path.join(willow.DATA_DIR, 'Duck', '060_0000.mat')
  mat_file2 = os.path.join(willow.DATA_DIR, 'Duck', '060_0001.mat')

  problem = make_matching_problem(
      mat_file1, mat_file2,
      n1=10, n2=30,
      shuffle=True, scale=0.2
  )
  print(problem)
  problem_directed = convert_to_directed(problem)
  print(problem_directed)


def _test_make_matching_problem_rng_helper(mat_file1, mat_file2,
                                           n1, n2,
                                           seed, n_problem):
  rng1 = np.random.default_rng(seed)
  rng2 = np.random.default_rng(seed)

  for _ in range(n_problem):
    problem1 = make_matching_problem(
        mat_file1,
        mat_file2,
        n1=n1,
        n2=n2,
        shuffle=True,
        scale=0.2,
        rng=rng1
    )
    problem2 = make_matching_problem(
        mat_file1,
        mat_file2,
        n1=n1,
        n2=n2,
        shuffle=True,
        scale=0.2,
        rng=rng2
    )
    assert_same_problems(problem1, problem2)


def test_make_matching_problem_rng():
  mat_file1 = os.path.join(willow.DATA_DIR, 'Duck', '060_0000.mat')
  mat_file2 = os.path.join(willow.DATA_DIR, 'Duck', '060_0001.mat')

  for seed in [101, 1024]:
    _test_make_matching_problem_rng_helper(mat_file1, mat_file2,
                                           n1=5, n2=5,
                                           seed=seed,
                                           n_problem=10)

    _test_make_matching_problem_rng_helper(mat_file1, mat_file2,
                                           n1=5, n2=10,
                                           seed=seed,
                                           n_problem=10)

    _test_make_matching_problem_rng_helper(mat_file1, mat_file2,
                                           n1=10, n2=10,
                                           seed=seed,
                                           n_problem=10)

    _test_make_matching_problem_rng_helper(mat_file1, mat_file2,
                                           n1=10, n2=30,
                                           seed=seed,
                                           n_problem=10)


if __name__ == '__main__':
  test_make_matching_problem()
  test_make_matching_problem_rng()
