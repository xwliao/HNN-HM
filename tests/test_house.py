import numpy as np

from utils.house import get_all_images
from utils.house import get_all_points


def test_get_all_images_same():
  images1 = get_all_images()
  images2 = get_all_images()
  for im1, im2 in zip(images1, images2):
    np.testing.assert_allclose(im1, im2)


def test_get_all_points_same():
  all_points1 = get_all_points()
  all_points2 = get_all_points()
  for p1, p2 in zip(all_points1, all_points2):
    np.testing.assert_allclose(p1, p2)


if __name__ == '__main__':
  test_get_all_images_same()
  test_get_all_points_same()
