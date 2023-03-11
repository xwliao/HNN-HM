import numpy as np
from pathlib import Path

from utils.utils import set_random_seed
from evaluate_house_helper import run


if __name__ == '__main__':
  key = 'gap'

  settings = {
      "gap": range(10, 110, 10),
      "nP1": 20,
      "nP2": 30,
      "scale": None,
      "shuffle_points": False,
      "shuffle_assignment": True
  }

  # xlabel = r'Gap'
  # text_list = [
  #     r'# of inliers $\itn_{\rmin}\rm$ = ' + '{}'.format(settings["nP1"]),
  #     r'# of outliers $\itn_{\rmout}\rm$ = ' + '{}'.format(settings["nP2"] - settings["nP1"]),
  #     r'Distance Scale = {}'.format(settings["scale"])
  # ]

  xlabel = r'Gap'
  text_list = []

  save_dir = Path('..') / 'results' / 'house' / 'gap'

  set_random_seed(1024)
  rng = np.random.default_rng(12345)

  run(settings, key=key, xlabel=xlabel, text_list=text_list,
      save_dir=save_dir, rng=rng)
