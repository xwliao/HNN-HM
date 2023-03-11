import numpy as np

from utils.make_matching_problem_house import make_matching_problem
from utils.make_matching_problem_house import convert_to_directed


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
  problem = make_matching_problem(
      index1=0, index2=110,
      nP1=10, nP2=30,
      shuffle_points=True,
      shuffle_assignment=True,
      scale=0.2
  )
  print(problem)
  problem_directed = convert_to_directed(problem)
  print(problem_directed)


def test_make_matching_problem_rng():
  seed = 1024
  rng1 = np.random.default_rng(seed)
  rng2 = np.random.default_rng(seed)

  for index1 in range(5):
    for index2 in range(10):
      problem1 = make_matching_problem(
          index1=index1,
          index2=index2,
          nP1=10,
          nP2=30,
          shuffle_points=True,
          shuffle_assignment=True,
          scale=0.2,
          rng=rng1
      )
      problem2 = make_matching_problem(
          index1=index1,
          index2=index2,
          nP1=10,
          nP2=30,
          shuffle_points=True,
          shuffle_assignment=True,
          scale=0.2,
          rng=rng2
      )
      assert_same_problems(problem1, problem2)


if __name__ == '__main__':
  test_make_matching_problem()
  test_make_matching_problem_rng()
