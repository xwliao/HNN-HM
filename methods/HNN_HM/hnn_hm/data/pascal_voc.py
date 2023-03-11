import numpy as np

from utils.make_matching_problem_pascal_voc import make_matching_problem
from utils.make_matching_problem_pascal_voc import count_common_points

from .data import BaseProblemGenerator


class PascalVOCProblemGenerator(BaseProblemGenerator):
  def __init__(self, *args, **kwargs):
    super(PascalVOCProblemGenerator, self).__init__(*args, **kwargs)

  def generate_problem(self, settings, rng=None):
    if rng is None:
      rng = self.default_rng
    rng = np.random.default_rng(rng)

    num_inlier_min = settings["num_inlier_min"]
    num_inlier_max = settings["num_inlier_max"]
    cropped = settings["cropped"]
    shuffle_points = settings["shuffle_points"]
    shuffle_assignment = settings["shuffle_assignment"]
    scale = settings["scale"]

    xml_file1 = settings.get('xml_file1', None)
    xml_file2 = settings.get('xml_file2', None)

    if (xml_file1 is None) or (xml_file2 is None):
      category = settings["category"]
      xml_files_all = settings["xml_files"]
      xml_files = xml_files_all[category]
      while True:
        idx1, idx2 = rng.choice(len(xml_files), size=2, replace=True)
        xml_file1 = xml_files[idx1]
        xml_file2 = xml_files[idx2]
        n_points = count_common_points(xml_file1, xml_file2)
        if n_points >= num_inlier_min:
          break
    else:
      n_points = count_common_points(xml_file1, xml_file2)
      assert n_points >= num_inlier_min

    problem = make_matching_problem(xml_file1, xml_file2,
                                    cropped=cropped,
                                    shuffle_points=shuffle_points,
                                    shuffle_assignment=shuffle_assignment,
                                    scale=scale,
                                    num_inlier_max=num_inlier_max,
                                    rng=rng)

    return problem
