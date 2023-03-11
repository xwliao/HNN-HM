import random
import numpy as np

from ADGM.adgm import adgm1
from ADGM.adgm import adgm2
from ADGM.adgm import adgm1_symmetric
from ADGM.adgm import adgm2_symmetric


def set_random_seed(seed):
  random.seed(seed)
  np.random.seed(seed)


def generate_problem(nP1, nP2, symmetric=False, rng=None):
  rng = np.random.default_rng(rng)

  N = nP2 * nP1

  if symmetric:
    indH1 = rng.choice(N, (10, 1), replace=True)
    valH1 = rng.random(size=(10, 1))

    # TODO: randomly generate
    indH2 = np.array([[0, 1], [1, 2], [2, 3],
                      [1, 0], [2, 1], [3, 2]], dtype=np.int32)
    valH2 = np.array([[0.1], [0.2], [0.3]] * 2, dtype=np.float64)

    indH3 = np.array([[0, 1, 2], [1, 2, 3], [2, 3, 4],
                      [0, 2, 1], [1, 3, 2], [2, 4, 3],
                      [1, 0, 2], [2, 1, 3], [3, 2, 4],
                      [1, 2, 0], [2, 3, 1], [3, 4, 2],
                      [2, 0, 1], [3, 1, 2], [4, 2, 3],
                      [2, 1, 0], [3, 2, 1], [4, 3, 2]], dtype=np.int32)
    valH3 = np.array([[0.1], [0.2], [0.3]] * 6, dtype=np.float64)
  else:
    indH1 = rng.choice(N, (10, 1), replace=True)
    valH1 = rng.random(size=(10, 1))

    indH2 = rng.choice(range(N), (20, 2), replace=True)
    valH2 = rng.random(size=(20, 1))

    indH3 = rng.choice(range(N), (30, 3), replace=True)
    valH3 = rng.random(size=(30, 1))

  problem = {
      "nP1": nP1,
      "nP2": nP2,
      "indH1": indH1,
      "valH1": valH1,
      "indH2": indH2,
      "valH2": valH2,
      "indH3": indH3,
      "valH3": valH3,
  }

  return problem


def test_adgm1():
  set_random_seed(seed=1024)
  rng = np.random.default_rng(101)

  nP1 = 10
  nP2 = 20
  problem = generate_problem(nP1, nP2, rng=rng)
  Xout = adgm1(problem)
  assert Xout.shape == (nP1, nP2)


def test_adgm2():
  set_random_seed(seed=1024)
  rng = np.random.default_rng(101)

  nP1 = 10
  nP2 = 20
  problem = generate_problem(nP1, nP2, rng=rng)
  Xout = adgm2(problem)
  assert Xout.shape == (nP1, nP2)


def test_adgm1_symmetric():
  set_random_seed(seed=1024)
  rng = np.random.default_rng(101)

  nP1 = 10
  nP2 = 20
  problem = generate_problem(nP1, nP2, symmetric=True, rng=rng)
  Xout = adgm1_symmetric(problem)
  assert Xout.shape == (nP1, nP2)


def test_adgm2_symmetric():
  set_random_seed(seed=1024)
  rng = np.random.default_rng(101)

  nP1 = 10
  nP2 = 20
  problem = generate_problem(nP1, nP2, symmetric=True, rng=rng)
  Xout = adgm2_symmetric(problem)
  assert Xout.shape == (nP1, nP2)


def generate_special_non_symmetric_problems():
  nP1 = 3
  nP2 = 3

  indH1 = None
  valH1 = None

  indH2 = None
  valH2 = None

  def case1():
    # Note: This is a bad case for the hypergraph matching problem,
    # because one point in one graph is appeared twice.
    # (a1, b3) => 2
    # (a3, b3) => 8
    # (a3, b1) => 6
    indH3 = np.array([[2, 8, 6]], dtype=np.int32)
    valH3 = np.array([[1.]], dtype=np.float64)
    return indH3, valH3

  def case2():
    indH3 = np.array([[4, 0, 2],
                      [1, 0, 6],
                      [0, 1, 3]],
                     dtype=np.int32)
    valH3 = np.array([[0.6],
                      [0.9],
                      [0.4]],
                     dtype=np.float64)
    return indH3, valH3

  def case3():
    indH3 = np.array([[0, 2, 4],
                      [0, 1, 6],
                      [0, 1, 6],
                      [0, 1, 3]],
                     dtype=np.int32)

    valH3 = np.array([[0.6],
                      [0.9],
                      [0.9],
                      [0.4]],
                     dtype=np.float64)
    return indH3, valH3

  def case4():
    indH3 = np.array([[1, 6, 0],
                      [0, 2, 0],
                      [1, 1, 7],
                      [8, 1, 7],
                      [0, 5, 8],
                      [5, 2, 2]],
                     dtype=np.int32)

    valH3 = np.array([[0.5],
                      [0.4],
                      [0.9],
                      [0.4],
                      [0.7],
                      [0.3]],
                     dtype=np.float64)
    return indH3, valH3

  for (indH3, valH3) in [case1(), case2(), case3(), case4()]:
    problem = {
        "nP1": nP1,
        "nP2": nP2,
        "indH1": indH1,
        "valH1": valH1,
        "indH2": indH2,
        "valH2": valH2,
        "indH3": indH3,
        "valH3": valH3,
    }
    yield problem


def _test_adgm_same_helper(func,
                           symmetric,
                           n_problem,
                           n_run,
                           nP1=10,
                           nP2=20,
                           seed=101,
                           seed_gloabal=1024):
  print(f'Func name: {func.__name__}')
  print(f'Symmetic: {symmetric}')

  assert n_run > 1
  set_random_seed(seed=seed_gloabal)
  rng = np.random.default_rng(seed)

  for _ in range(n_problem):
    problem = generate_problem(nP1, nP2, symmetric=symmetric, rng=rng)
    Xouts = [func(problem) for _ in range(n_run)]
    for Xout in Xouts[1:]:
      np.testing.assert_allclose(Xout, Xouts[0])

  if not symmetric:
    for problem in generate_special_non_symmetric_problems():
      Xouts = [func(problem) for _ in range(n_run)]
      for Xout in Xouts[1:]:
        np.testing.assert_allclose(Xout, Xouts[0])


def test_adgm_same():
  _test_adgm_same_helper(func=adgm1,
                         symmetric=False,
                         n_problem=200,
                         n_run=2,
                         nP1=10,
                         nP2=20)

  _test_adgm_same_helper(func=adgm1,
                         symmetric=True,
                         n_problem=200,
                         n_run=2,
                         nP1=10,
                         nP2=20)

  _test_adgm_same_helper(func=adgm2,
                         symmetric=False,
                         n_problem=200,
                         n_run=2,
                         nP1=10,
                         nP2=20)

  _test_adgm_same_helper(func=adgm2,
                         symmetric=True,
                         n_problem=200,
                         n_run=2,
                         nP1=10,
                         nP2=20)

  _test_adgm_same_helper(func=adgm1_symmetric,
                         symmetric=True,
                         n_problem=200,
                         n_run=2,
                         nP1=10,
                         nP2=20)

  _test_adgm_same_helper(func=adgm2_symmetric,
                         symmetric=True,
                         n_problem=200,
                         n_run=2,
                         nP1=10,
                         nP2=20)


def main():
  test_adgm1()
  test_adgm2()
  test_adgm1_symmetric()
  test_adgm2_symmetric()
  test_adgm_same()


if __name__ == '__main__':
  main()
