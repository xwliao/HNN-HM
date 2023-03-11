import random
import numpy as np
from utils import spair


def set_random_seed(seed):
  random.seed(seed)
  np.random.seed(seed)


def _get_keypoints(keypoint_list, dtype=None):
  points = [[p["x"], p["y"]] for p in keypoint_list]
  points = np.asarray(points, dtype=dtype)
  return points


def _test_random_helper(sets, num_samples):
  set_random_seed(seed=1024)

  rng = np.random.default_rng(12345)
  dataset1 = spair.get_dataset(sets, rng=rng)

  rng = np.random.default_rng(12345)
  dataset2 = spair.get_dataset(sets, rng=rng)

  for _ in range(num_samples):
    ((anno1a, anno1b), assignmentMatrix1) = dataset1.get_k_samples(idx=None, k=2, mode="intersection")
    ((anno2a, anno2b), assignmentMatrix2) = dataset2.get_k_samples(idx=None, k=2, mode="intersection")

    dtype = np.float64
    points1a = _get_keypoints(anno1a["keypoints"], dtype=dtype)
    points1b = _get_keypoints(anno1b["keypoints"], dtype=dtype)

    points2a = _get_keypoints(anno2a["keypoints"], dtype=dtype)
    points2b = _get_keypoints(anno2b["keypoints"], dtype=dtype)

    np.testing.assert_allclose(points1a, points2a)
    np.testing.assert_allclose(points1b, points2b)
    np.testing.assert_allclose(assignmentMatrix1, assignmentMatrix2)


def test_random():
  _test_random_helper("train", 100)
  _test_random_helper("val", 100)
  _test_random_helper("test", 100)


if __name__ == '__main__':
  test_random()
