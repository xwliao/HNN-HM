import os
import numpy as np

from utils import pascal_voc

from .config import BaseConfig


class PascalVOCConfig(BaseConfig):
  def __init__(self):
    super(PascalVOCConfig, self).__init__()

    self.DATASET_NAME = "Pascal VOC"

    self.set_result_dir(name="pascal_voc")

    self.NORMALIZE_POINT_SET = True

    self.NUM_ITERATIONS_TR = 300000
    self.BATCH_SIZE_TR = 16

    self.SETTINGS_TR = {
        "xml_file1": None,
        "xml_file2": None,
        "num_inlier_min": pascal_voc.MIN_NUM_KEYPOINTS,
        "num_inlier_max": np.arange(3, 13).tolist(),
        "cropped": True,
        "shuffle_points": True,
        "shuffle_assignment": True,
        "scale": None,
        "category": list(pascal_voc.CATEGORIES),
        "xml_files": pascal_voc.get_all_xml_files(pascal_voc.TRAIN_LIST_DIR,
                                                  min_num_keypoints=pascal_voc.MIN_NUM_KEYPOINTS),
    }

    self.SETTINGS_GE = {
        "xml_file1": None,
        "xml_file2": None,
        "num_inlier_min": pascal_voc.MIN_NUM_KEYPOINTS,
        "num_inlier_max": -1,
        "cropped": True,
        "shuffle_points": False,
        "shuffle_assignment": True,
        "scale": None,
        "category": list(pascal_voc.CATEGORIES),
        "xml_files": pascal_voc.get_all_xml_files(pascal_voc.TRAIN_LIST_DIR,
                                                  min_num_keypoints=pascal_voc.MIN_NUM_KEYPOINTS),
    }

    self.SETTINGS_SIMPLE = {
        "xml_file1": os.path.join('aeroplane', '2008_008467_1.xml'),
        "xml_file2": os.path.join('aeroplane', '2009_005210_1.xml'),
        "num_inlier_min": 3,
        "num_inlier_max": -1,
        "cropped": True,
        "shuffle_points": False,
        "shuffle_assignment": False,
        "scale": None
    }
