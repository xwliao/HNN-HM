import random
import numpy as np

from TM.tensor_matching import tensor_matching


def set_random_seed(seed):
  random.seed(seed)
  np.random.seed(seed)


def generate_problem(nP1, nP2):
  N = nP2 * nP1
  indH1 = np.random.choice(range(N), (10, 1))
  valH1 = np.random.rand(10, 1)

  indH2 = np.random.choice(range(N), (20, 2))
  valH2 = np.random.rand(20, 1)

  indH3 = np.random.choice(range(N), (30, 3))
  valH3 = np.random.rand(30, 1)

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


def test_tensor_matching():
  set_random_seed(seed=1024)

  nP1 = 10
  nP2 = 20

  problem = generate_problem(nP1, nP2)

  Xout = tensor_matching(problem)
  assert Xout.shape == (nP1, nP2)

  Xout = tensor_matching(problem, max_iter=100, sparse=True, stoc='default')
  assert Xout.shape == (nP1, nP2)

  Xout = tensor_matching(problem, max_iter=100, sparse=True, stoc='single')
  assert Xout.shape == (nP1, nP2)

  Xout = tensor_matching(problem, max_iter=100, sparse=True, stoc='doubly')
  assert Xout.shape == (nP1, nP2)


def test_tensor_matching_same():
  set_random_seed(seed=1024)

  nP1 = 10
  nP2 = 20

  problem = generate_problem(nP1, nP2)

  Xout1 = tensor_matching(problem)
  Xout2 = tensor_matching(problem)
  np.testing.assert_allclose(Xout1, Xout2)

  Xout1 = tensor_matching(problem, max_iter=100, sparse=True, stoc='default')
  Xout2 = tensor_matching(problem, max_iter=100, sparse=True, stoc='default')
  np.testing.assert_allclose(Xout1, Xout2)

  Xout1 = tensor_matching(problem, max_iter=100, sparse=True, stoc='single')
  Xout2 = tensor_matching(problem, max_iter=100, sparse=True, stoc='single')
  np.testing.assert_allclose(Xout1, Xout2)

  Xout1 = tensor_matching(problem, max_iter=100, sparse=True, stoc='doubly')
  Xout2 = tensor_matching(problem, max_iter=100, sparse=True, stoc='doubly')
  np.testing.assert_allclose(Xout1, Xout2)


def main():
  test_tensor_matching()
  test_tensor_matching_same()


if __name__ == '__main__':
  main()
