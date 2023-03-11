import os
import numpy as np

from utils import willow

from .config import BaseConfig


class WillowConfig(BaseConfig):
  def __init__(self):
    super(WillowConfig, self).__init__()

    self.DATASET_NAME = "Willow"

    self.set_result_dir(name="willow")

    self.NORMALIZE_POINT_SET = True

    self.NUM_ITERATIONS_TR = 100000
    self.BATCH_SIZE_TR = 4

    self.SETTINGS_TR = {
        "mat_file1": None,
        "mat_file2": None,
        "n1": 10,
        "n2": np.arange(10, 20).tolist(),
        "shuffle": True,
        "scale": None,
        "category": list(willow.CATEGORIES),
        "mat_files": {category: willow.get_mat_files_train(category) for category in willow.CATEGORIES}
    }

    self.SETTINGS_GE = {
        "xml_file1": None,
        "xml_file2": None,
        "n1": 10,
        "n2": np.arange(10, 20).tolist(),
        "shuffle": True,
        "scale": None,
        "category": list(willow.CATEGORIES),
        "mat_files": {category: willow.get_mat_files_train(category) for category in willow.CATEGORIES}
    }

    self.SETTINGS_SIMPLE = {
        "mat_file1": os.path.join(willow.DATA_DIR, 'Duck', '060_0000.mat'),
        "mat_file2": os.path.join(willow.DATA_DIR, 'Duck', '060_0001.mat'),
        "n1": 10,
        "n2": 10,
        "shuffle": True,
        "scale": None
    }
