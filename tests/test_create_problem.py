import numbers
import numpy as np

from utils.create_problem import create_problem
from utils.create_problem import convert_to_directed


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


def test_create_problem():
  rng = np.random.default_rng(1234)
  nP1 = 10
  nP2 = 20
  P1 = rng.random(size=(nP1, 2))
  P2 = rng.random(size=(nP2, 2))
  assignmentMatrix = np.eye(nP1, nP2)
  scale = 1.

  for order in [2, 3, (2, 3), [2, 3]]:
    problem = create_problem(P1, P2,
                             assignmentMatrix,
                             scale=scale,
                             order=order,
                             rng=rng)
    print(problem)


def test_create_problem_3rd_scale_none():
  rng = np.random.default_rng(1234)
  nP1 = 10
  nP2 = 20
  P1 = rng.random(size=(nP1, 2))
  P2 = rng.random(size=(nP2, 2))
  assignmentMatrix = np.eye(nP1, nP2)
  order = 3

  problem = create_problem(P1, P2,
                           assignmentMatrix,
                           scale=1.,
                           order=order,
                           rng=rng)
  print(problem)
  assert problem["indH3"] is not None
  assert problem["valH3"] is not None

  problem = create_problem(P1, P2,
                           assignmentMatrix,
                           scale=None,
                           order=order,
                           rng=rng)
  print(problem)
  assert problem["indH3"] is not None
  assert problem["valH3"] is not None

  problem = create_problem(P1, P2,
                           assignmentMatrix,
                           order=order,
                           rng=rng)
  print(problem)
  assert problem["indH3"] is not None
  assert problem["valH3"] is not None


def test_create_problem_rng():
  seed = 1024
  rng1 = np.random.default_rng(seed)
  rng2 = np.random.default_rng(seed)

  rng = np.random.default_rng(1234)
  nP1 = 10
  nP2 = 20
  P1 = rng.random(size=(nP1, 2))
  P2 = rng.random(size=(nP2, 2))
  assignmentMatrix = np.eye(nP1, nP2)
  scale = 1.

  for _ in range(10):
    for order in [2, 3, (2, 3), [2, 3]]:
      problem1 = create_problem(P1, P2,
                                assignmentMatrix,
                                scale=scale,
                                order=order,
                                rng=rng1)
      problem2 = create_problem(P1, P2,
                                assignmentMatrix,
                                scale=scale,
                                order=order,
                                rng=rng2)
      assert_same_problems(problem1, problem2)


def test_convert_to_directed():
  rng = np.random.default_rng(1234)
  nP1 = 10
  nP2 = 20
  P1 = rng.random(size=(nP1, 2))
  P2 = rng.random(size=(nP2, 2))
  assignmentMatrix = np.eye(nP1, nP2)
  scale = 1.

  for order in [2, 3, (2, 3), [2, 3]]:
    problem = create_problem(P1, P2,
                             assignmentMatrix,
                             scale=scale,
                             order=order,
                             rng=rng)
    problem_directed = convert_to_directed(problem, copy=True)
    print(problem)
    print(problem_directed)


def test_convert_to_directed_unique_and_sort():
  indH2 = np.array([[3, 2],
                    [0, 1]],
                   dtype=np.int32)

  valH2 = np.array([[1],
                    [0]],
                   dtype=np.float64)

  indH2_t = np.array([[0, 1],   # 0
                      [1, 0],   # 0
                      [2, 3],   # 1
                      [3, 2]],  # 1
                     dtype=np.int32)

  valH2_t = np.array([[0],
                      [0],
                      [1],
                      [1]],
                     dtype=np.float64)

  indH3 = np.array([[3, 2, 1],
                    [0, 1, 2]],
                   dtype=np.int32)

  valH3 = np.array([[1],
                    [0]],
                   dtype=np.float64)

  indH3_t = np.array([[0, 1, 2],   # 0
                      [0, 2, 1],   # 0
                      [1, 0, 2],   # 0
                      [1, 2, 0],   # 0
                      [1, 2, 3],   # 1
                      [1, 3, 2],   # 1
                      [2, 0, 1],   # 0
                      [2, 1, 0],   # 0
                      [2, 1, 3],   # 1
                      [2, 3, 1],   # 1
                      [3, 1, 2],   # 1
                      [3, 2, 1]],  # 1
                     dtype=np.int32)

  valH3_t = np.array([[0],
                      [0],
                      [0],
                      [0],
                      [1],
                      [1],
                      [0],
                      [0],
                      [1],
                      [1],
                      [1],
                      [1]],
                     dtype=np.float64)

  problem = {
      "indH1": None,
      "valH1": None,
      "indH2": indH2,
      "valH2": valH2,
      "indH3": indH3,
      "valH3": valH3
  }

  problem_directed = convert_to_directed(problem,
                                         unique_and_sort=True,
                                         copy=True)

  np.testing.assert_allclose(problem_directed["indH2"], indH2_t)
  np.testing.assert_allclose(problem_directed["valH2"], valH2_t)
  np.testing.assert_allclose(problem_directed["indH3"], indH3_t)
  np.testing.assert_allclose(problem_directed["valH3"], valH3_t)


def test_convert_to_directed_copy():
  rng = np.random.default_rng(1234)
  nP1 = 10
  nP2 = 20
  P1 = rng.random(size=(nP1, 2))
  P2 = rng.random(size=(nP2, 2))
  assignmentMatrix = np.eye(nP1, nP2)
  scale = 1.

  for order in [2, 3, (2, 3), [2, 3]]:
    problem = create_problem(P1, P2,
                             assignmentMatrix,
                             scale=scale,
                             order=order,
                             rng=rng)
    problem_directed = convert_to_directed(problem, copy=True)
    for (k, v) in problem_directed.items():
      if v is None:
        assert problem[k] is None
      elif not isinstance(v, numbers.Number):
        print(f"Checking the deepcopy of key {k}")
        print(f'Type {type(v)}')
        assert v is not problem[k], f'Different in {k}'


def test_convert_to_directed_same():
  rng = np.random.default_rng(1234)
  nP1 = 10
  nP2 = 20
  P1 = rng.random(size=(nP1, 2))
  P2 = rng.random(size=(nP2, 2))
  assignmentMatrix = np.eye(nP1, nP2)
  scale = 1.

  for order in [2, 3, (2, 3), [2, 3]]:
    problem = create_problem(P1, P2,
                             assignmentMatrix,
                             scale=scale,
                             order=order,
                             rng=rng)
    problem_directed_1 = convert_to_directed(problem)
    problem_directed_2 = convert_to_directed(problem)
    assert_same_problems(problem_directed_1, problem_directed_2)


if __name__ == '__main__':
  test_create_problem()
  test_create_problem_3rd_scale_none()
  test_create_problem_rng()
  test_convert_to_directed()
  test_convert_to_directed_unique_and_sort()
  test_convert_to_directed_copy()
  test_convert_to_directed_same()
