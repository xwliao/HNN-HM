import numpy as np
from pathlib import Path

from utils.utils import set_random_seed
from evaluate_house_helper import run


if __name__ == '__main__':
  key = 'nOutlier'

  settings = {
      "gap": 50,
      "nOutlier": range(0, 11),
      "nP2": 30,
      "scale": None,
      "shuffle_points": False,
      "shuffle_assignment": True
  }

  # xlabel = r'# of outliers'
  # text_list = [
  #     r'Gap = {}'.format(settings["gap"]),
  #     r'Distance Scale = {}'.format(settings["scale"])
  # ]

  xlabel = r'# of outliers'
  text_list = []

  save_dir = Path('..') / 'results' / 'house' / 'outlier'

  set_random_seed(1024)
  rng = np.random.default_rng(12345)

  run(settings, key=key, xlabel=xlabel, text_list=text_list,
      save_dir=save_dir, rng=rng)
