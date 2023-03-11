import numpy as np

from .create_problem import create_problem
from .create_problem import convert_to_directed  # noqa


def _get_keypoints(keypoint_list, dtype=None):
  points = [[p["x"], p["y"]] for p in keypoint_list]
  points = np.asarray(points, dtype=dtype)
  return points


def make_matching_problem(data, scale, rng=None):
  """
  Output:
    problem: dict
    {
      nP1:   int
      nP2:   int
      P1:    float64, (nP1, 2)
      P2:    float64, (nP2, 2)
      indH1: int32,   (N1, 1) array or None
      valH1: float64, (N1,)   array or None
      indH2: int32,   (N2, 2) array or None
      valH2: float64, (N2,)   array or None
      indH3: int32,   (N3, 3) array or None
      valH3: float64, (N3,)   array or None
      image1: uint8, (height, width, channel) array or None
      image2: uint8, (height, width, channel) array or None
      assignmentMatrix: bool, (nP1, nP2)
    }
  """
  ((anno1, anno2), assignmentMatrix) = data

  point_dtype = np.float64
  P1 = _get_keypoints(anno1["keypoints"], dtype=point_dtype)
  P2 = _get_keypoints(anno2["keypoints"], dtype=point_dtype)

  image_dtype = np.uint8
  image1 = np.asarray(anno1["image"], dtype=image_dtype)
  image2 = np.asarray(anno2["image"], dtype=image_dtype)

  problem = create_problem(P1, P2,
                           assignmentMatrix,
                           scale=scale,
                           image1=image1, image2=image2,
                           order=3,
                           rng=rng)

  return problem
