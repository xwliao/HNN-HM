import numpy as np
from utils.eval_hypergraph_matching import evaluate


def get_data(name, dtype=None):
  return np.loadtxt('test_data/{}.txt'.format(name), dtype=dtype)


def test_evaluate_data_from_matlab_code():
  X = get_data('X').T  # nP1 x nP2

  problem = {}
  assignmentMatrix = get_data('assignmentMatrix', dtype=np.bool)
  problem["assignmentMatrix"] = assignmentMatrix
  problem["indH3"] = get_data('indH3', dtype=np.int32)
  problem["valH3"] = get_data('valH3')

  accuracy, match_score = evaluate(problem, X)

  accuracy_target = get_data('accuracy')
  match_score_target = get_data('MatchSocre')

  np.testing.assert_allclose(accuracy, accuracy_target, rtol=1e-5, atol=0)
  np.testing.assert_allclose(match_score, match_score_target, rtol=1e-5, atol=0)


def test_evaluate():
  # TODO: Manually create test cases.
  pass


if __name__ == '__main__':
  test_evaluate_data_from_matlab_code()
  test_evaluate()
