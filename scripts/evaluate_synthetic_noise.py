import numpy as np
from pathlib import Path

from utils.utils import set_random_seed
from evaluate_synthetic_helper import run


if __name__ == '__main__':
  n_test = 100

  key = 'deformation'

  settings_template = {
      "nInlier": 20,
      "nOutlier": 0,
      "deformation": np.arange(0, 0.225, 0.025).tolist(),
      "typeDistribution": 'normal',  # normal / uniform
      "transScale": 1.0,   # scale change
      "transRotate": 0.0,  # rotation change
      "scale": None,
      "bPermute": True
  }

  # xlabel = r'Deformation noise $\it\sigma\rm$'
  # text_list = [
  #     r'# of inliers $\itn_{\rmin}\rm$ = ' + '{}'.format(settings_template["nInlier"]),
  #     r'# of outliers $\itn_{\rmout}\rm$ = ' + '{}'.format(settings_template["nOutlier"])
  # ]

  xlabel = r'Noise level'
  text_list = []

  save_dir = Path('..') / 'results' / 'synthetic' / 'noise'

  set_random_seed(1024)
  rng = np.random.default_rng(12345)

  run(n_test, settings_template, key=key, xlabel=xlabel, text_list=text_list,
      save_dir=save_dir, rng=rng)
