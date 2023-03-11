import numpy as np

from .config import BaseConfig


class SyntheticConfig(BaseConfig):
  def __init__(self):
    super(SyntheticConfig, self).__init__()

    self.DATASET_NAME = "Synthetic"

    self.set_result_dir(name="synthetic")

    self.NUM_ITERATIONS_TR = 100000
    self.BATCH_SIZE_TR = 4

    self.SETTINGS_TR = {
        # "nInlier": 20,  # TODO
        # "nInlier": 4,  # TODO
        "nInlier": np.arange(5, 9).tolist(),  # TODO
        "nOutlier": np.arange(15).tolist(),
        "deformation": np.arange(0, 0.225, 0.025).tolist(),
        "typeDistribution": 'normal',  # normal / uniform
        "transScale": np.arange(0.9, 1.7, 0.1).tolist(),   # scale change
        "transRotate": 0.0,  # rotation change
        "scale": None,
        "bPermute": True
    }

    self.SETTINGS_GE = {
        "nInlier": 20,  # TODO
        # "nInlier": 4,  # TODO
        # "nInlier": np.arange(9, 13).tolist(),  # TODO
        "nOutlier": np.arange(11).tolist(),
        "deformation": np.arange(0, 0.225, 0.025).tolist(),
        "typeDistribution": 'normal',  # normal / uniform
        "transScale": [1.0, 1.1, 1.5],   # scale change
        "transRotate": 0.0,  # rotation change
        "scale": None,
        "bPermute": True
    }

    self.SETTINGS_SIMPLE = {
        "nInlier": 4,
        "nOutlier": 0,
        "deformation": 0,
        "typeDistribution": 'normal',  # normal / uniform
        "transScale": 1.0,   # scale change
        "transRotate": 0.0,  # rotation change
        "scale": None,
        "bPermute": True
    }
