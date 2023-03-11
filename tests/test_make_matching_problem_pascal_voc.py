import os
import random
import numpy as np
from utils.make_matching_problem_pascal_voc import make_matching_problem
from utils.make_matching_problem_pascal_voc import convert_to_directed


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

  xml_file1 = os.path.join('aeroplane', '2008_008467_1.xml')
  xml_file2 = os.path.join('aeroplane', '2009_005210_1.xml')

  problem = make_matching_problem(
      xml_file1=xml_file1, xml_file2=xml_file2,
      cropped=True, shuffle_points=True, shuffle_assignment=True, scale=0.2
  )
  print(problem)
  problem_directed = convert_to_directed(problem)
  print(problem_directed)


def test_make_matching_problem_same():
  set_random_seed(seed=1024)

  xml_file1 = os.path.join('aeroplane', '2008_008467_1.xml')
  xml_file2 = xml_file1

  problem = make_matching_problem(
      xml_file1=xml_file1, xml_file2=xml_file2,
      cropped=False, shuffle_points=False, shuffle_assignment=False, scale=0.2
  )
  print(problem)
  problem_directed = convert_to_directed(problem)
  print(problem_directed)

  X = problem["assignmentMatrix"]
  X_tgt = np.eye(X.shape[0], dtype=X.dtype)
  np.testing.assert_allclose(X, X_tgt)

  P1 = problem["P1"]
  P2 = problem["P2"]
  np.testing.assert_allclose(P1, P2)


def test_make_matching_problem_same_cropped():
  set_random_seed(seed=1024)

  xml_file1 = os.path.join('aeroplane', '2008_008467_1.xml')
  xml_file2 = xml_file1

  problem = make_matching_problem(
      xml_file1=xml_file1, xml_file2=xml_file2,
      cropped=True, shuffle_points=False, shuffle_assignment=False, scale=0.2
  )

  X = problem["assignmentMatrix"]
  X_tgt = np.eye(X.shape[0], dtype=X.dtype)
  np.testing.assert_allclose(X, X_tgt)

  P1 = problem["P1"]
  P2 = problem["P2"]
  np.testing.assert_allclose(P1, P2)


def test_make_matching_problem_same_shuffle():
  set_random_seed(seed=1024)

  xml_file1 = os.path.join('aeroplane', '2008_008467_1.xml')
  xml_file2 = xml_file1

  problem = make_matching_problem(
      xml_file1=xml_file1, xml_file2=xml_file2,
      cropped=False, shuffle_points=True, shuffle_assignment=True, scale=0.2
  )
  print(problem)
  problem_directed = convert_to_directed(problem)
  print(problem_directed)

  P1 = problem["P1"]
  P2 = problem["P2"]
  assert P1.shape == P2.shape

  ind = np.argmax(problem["assignmentMatrix"], axis=1)
  P = P2[ind]
  np.testing.assert_allclose(P, P1)


def test_make_matching_problem_same_shuffle_cropped():
  set_random_seed(seed=1024)

  xml_file1 = os.path.join('aeroplane', '2008_008467_1.xml')
  xml_file2 = xml_file1

  problem = make_matching_problem(
      xml_file1=xml_file1, xml_file2=xml_file2,
      cropped=True, shuffle_points=True, shuffle_assignment=True, scale=0.2
  )
  print(problem)
  problem_directed = convert_to_directed(problem)
  print(problem_directed)

  P1 = problem["P1"]
  P2 = problem["P2"]
  assert P1.shape == P2.shape

  ind = np.argmax(problem["assignmentMatrix"], axis=1)
  P = P2[ind]
  np.testing.assert_allclose(P, P1)


def test_make_matching_problem_rng():
  seed = 1024
  rng1 = np.random.default_rng(seed)
  rng2 = np.random.default_rng(seed)

  xml_file1 = os.path.join('aeroplane', '2008_008467_1.xml')
  xml_file2 = os.path.join('aeroplane', '2009_005210_1.xml')

  for _ in range(10):
    problem1 = make_matching_problem(
        xml_file1=xml_file1,
        xml_file2=xml_file2,
        cropped=True,
        shuffle_points=True,
        shuffle_assignment=True,
        scale=0.2,
        rng=rng1
    )
    problem2 = make_matching_problem(
        xml_file1=xml_file1,
        xml_file2=xml_file2,
        cropped=True,
        shuffle_points=True,
        shuffle_assignment=True,
        scale=0.2,
        rng=rng2
    )
    assert_same_problems(problem1, problem2)


if __name__ == '__main__':
  test_make_matching_problem()
  test_make_matching_problem_same()
  test_make_matching_problem_same_cropped()
  test_make_matching_problem_same_shuffle()
  test_make_matching_problem_same_shuffle_cropped()
  test_make_matching_problem_rng()
