import numpy as np
from pathlib import Path

from utils.utils import set_random_seed
from evaluate_synthetic_helper import run


if __name__ == '__main__':
  n_test = 100

  key = 'nOutlier'

  nOutlier_lists = np.arange(0, 110, 10)

  settings_template = {
      "nInlier": 10,
      "nOutlier": nOutlier_lists,
      "deformation": 0.03,
      "typeDistribution": 'normal',  # normal / uniform
      "transScale": 1.5,   # scale change
      "transRotate": 0.0,  # rotation change
      "scale": None,
      "bPermute": True
  }

  # xlabel = r'# of outliers $\itn_{\rmout}\rm$'
  # text_list = [
  #     '# of tests = {}'.format(n_test),
  #     '# of inliers {} = {}'.format(r'$\itn_{\rmin}\rm$', settings_template["nInlier"]),
  #     'Deformation noise {} = {}'.format(r'$\it\sigma\rm$', settings_template["deformation"]),
  #     'Transformation Scale = {}'.format(settings_template["transScale"]),
  #     'Distance Scale = {}'.format(settings_template["scale"])
  # ]

  xlabel = r'# of outliers'
  text_list = []

  save_dir = Path('..') / 'results' / 'synthetic' / 'outlier'

  set_random_seed(1024)
  rng = np.random.default_rng(12345)

  run(n_test, settings_template, key=key, xlabel=xlabel, text_list=text_list, save_dir=save_dir, rng=rng)
