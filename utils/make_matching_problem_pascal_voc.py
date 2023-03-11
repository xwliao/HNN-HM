import numpy as np

from . import pascal_voc
from .create_problem import create_problem
from .create_problem import convert_to_directed  # noqa


def count_common_points(xml_file1, xml_file2):
  _, names1 = pascal_voc.get_keypoints(xml_file1)
  _, names2 = pascal_voc.get_keypoints(xml_file2)
  same_names = list(set(names1) & set(names2))
  return len(same_names)


def get_cid(xml_file):
  category = pascal_voc.get_category(xml_file)
  cid = pascal_voc.CATEGORIES.index(category)
  return cid


def _generate_point_sets(xml_file1, xml_file2, cropped, shuffle_points, shuffle_assignment,
                         num_inlier_max=-1, num_outlier_max=0, rng=None):
  """
  ```python
  assert npts1 == npts2
  if num_inlier_max < 0:
    n1 = npts1
    n2 = npts2
  else:
    assert num_outlier_max >= 0
    n1 = min(npts1, num_inlier_max)
    n2 = min(npts2, n1 + num_outlier_max)
  ```

  Note: Since npts1 == npts2, we have n2 >= n1
  """
  rng = np.random.default_rng(rng)

  pts1, names1 = pascal_voc.get_keypoints(xml_file1, cropped=cropped)
  pts2, names2 = pascal_voc.get_keypoints(xml_file2, cropped=cropped)

  same_names = list(set(names1) & set(names2))
  same_names.sort()  # in order to reproduce the experimental results stably

  if shuffle_points:
      inds = rng.permutation(len(same_names))
      same_names = [same_names[idx] for idx in inds]

  inds1 = np.asarray([names1.index(name) for name in same_names])
  inds2 = np.asarray([names2.index(name) for name in same_names])

  npts1 = len(inds1)
  npts2 = len(inds2)
  assert npts1 == npts2

  if num_inlier_max < 0:
    n1 = npts1
    n2 = npts2
  else:
    assert num_outlier_max >= 0
    n1 = min(npts1, num_inlier_max)
    n2 = min(npts2, n1 + num_outlier_max)

  if shuffle_assignment:
    perm1 = rng.permutation(n1)
    perm2 = rng.permutation(n2)
  else:
    perm1 = np.arange(n1)
    perm2 = np.arange(n2)

  pts1 = pts1[inds1[perm1]]
  pts2 = pts2[inds2[perm2]]

  assignmentMatrix = np.eye(n1, n2, dtype=np.bool)
  assignmentMatrix = assignmentMatrix[perm1]
  assignmentMatrix = assignmentMatrix[:, perm2]

  return pts1, pts2, assignmentMatrix


def make_matching_problem(xml_file1, xml_file2, cropped, shuffle_points, shuffle_assignment, scale,
                          num_inlier_max=-1, num_outlier_max=0,
                          rng=None):
  """
  Output:
    problem: dict
    {
      cid:   int
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
  rng = np.random.default_rng(rng)

  cid1 = get_cid(xml_file1)
  cid2 = get_cid(xml_file2)
  assert cid1 == cid2
  cid = cid1

  P1, P2, assignmentMatrix = _generate_point_sets(xml_file1, xml_file2, cropped=cropped,
                                                  shuffle_points=shuffle_points,
                                                  shuffle_assignment=shuffle_assignment,
                                                  num_inlier_max=num_inlier_max,
                                                  num_outlier_max=num_outlier_max,
                                                  rng=rng)

  image1 = pascal_voc.get_image(xml_file1, cropped=cropped)
  image2 = pascal_voc.get_image(xml_file2, cropped=cropped)

  problem = create_problem(P1, P2,
                           assignmentMatrix,
                           scale=scale,
                           image1=image1, image2=image2,
                           cid=cid,
                           order=3,
                           rng=rng)

  return problem
