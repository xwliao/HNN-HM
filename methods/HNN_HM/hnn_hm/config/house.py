import numpy as np

from .config import BaseConfig


class HouseConfig(BaseConfig):
  def __init__(self):
    super(HouseConfig, self).__init__()

    self.DATASET_NAME = "House"

    self.set_result_dir(name="house")

    self.NORMALIZE_POINT_SET = False

    self.NUM_ITERATIONS_TR = 100000
    self.BATCH_SIZE_TR = 3

    self.SETTINGS_TR = {
        "nP1": np.arange(4, 11).tolist(),
        "nP2": np.arange(10, 21).tolist(),
        "indices": np.arange(0, 111, 5),
        "gap": np.arange(0, 23).tolist(),
        "scale": None,
        "shuffle_points": True,
        "shuffle_assignment": True
    }

    self.SETTINGS_GE = {
        "nP1": np.arange(10, 31).tolist(),
        "nP2": 30,
        "indices": np.arange(0, 111, 5),
        "gap": np.arange(0, 23).tolist(),
        "scale": None,
        "shuffle_points": False,
        "shuffle_assignment": True
    }

    self.SETTINGS_SIMPLE = {
        "nP1": 4,
        "nP2": 4,
        "indices": np.arange(0, 1),
        "gap": 0,
        "scale": None,
        "shuffle_points": False,
        "shuffle_assignment": False
    }
