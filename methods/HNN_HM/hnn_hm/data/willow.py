import numpy as np

from utils.make_matching_problem_willow import make_matching_problem

from .data import BaseProblemGenerator


class WillowProblemGenerator(BaseProblemGenerator):
  def __init__(self, *args, **kwargs):
    super(WillowProblemGenerator, self).__init__(*args, **kwargs)

  def generate_problem(self, settings, rng=None):
    if rng is None:
      rng = self.default_rng
    rng = np.random.default_rng(rng)

    n1 = settings["n1"]
    n2 = settings["n2"]
    shuffle = settings["shuffle"]
    scale = settings["scale"]

    mat_file1 = settings.get('mat_file1', None)
    mat_file2 = settings.get('mat_file2', None)

    if (mat_file1 is None) or (mat_file2 is None):
      category = settings["category"]
      mat_files_all = settings["mat_files"]
      mat_files = mat_files_all[category]
      index1, index2 = rng.choice(len(mat_files), size=2, replace=True)
      mat_file1 = mat_files[index1]
      mat_file2 = mat_files[index2]

    problem = make_matching_problem(mat_file1, mat_file2,
                                    n1=n1, n2=n2,
                                    shuffle=shuffle,
                                    scale=scale,
                                    rng=rng)

    return problem
