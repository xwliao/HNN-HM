from utils.make_matching_problem_synthetic import make_matching_problem

from .data import BaseProblemGenerator


class SyntheticProblemGenerator(BaseProblemGenerator):
  def __init__(self, *args, **kwargs):
    super(SyntheticProblemGenerator, self).__init__(*args, **kwargs)

  def generate_problem(self, settings, rng=None):
    if rng is None:
      rng = self.default_rng
    problem = make_matching_problem(settings, rng=rng)
    return problem
