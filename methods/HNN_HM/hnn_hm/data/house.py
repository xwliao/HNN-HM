import numpy as np

from utils.make_matching_problem_house import make_matching_problem

from .data import BaseProblemGenerator


class HouseProblemGenerator(BaseProblemGenerator):
  def __init__(self, *args, **kwargs):
    super(HouseProblemGenerator, self).__init__(*args, **kwargs)

  def generate_problem(self, settings, rng=None):
    if rng is None:
      rng = self.default_rng
    rng = np.random.default_rng(rng)

    nP1 = settings["nP1"]
    nP2 = settings["nP2"]
    indices = settings["indices"]
    gap = settings["gap"]
    shuffle_points = settings["shuffle_points"]
    shuffle_assignment = settings["shuffle_assignment"]
    scale = settings["scale"]

    ind1 = rng.integers(0, len(indices) - gap)
    ind2 = ind1 + gap

    if shuffle_assignment:
      # Randomly swap ind1 and ind2
      ind1, ind2 = rng.permutation([ind1, ind2])

    index1 = indices[ind1]
    index2 = indices[ind2]
    problem = make_matching_problem(index1, index2, nP1, nP2,
                                    shuffle_points=shuffle_points,
                                    shuffle_assignment=shuffle_assignment,
                                    scale=scale,
                                    rng=rng)
    return problem
