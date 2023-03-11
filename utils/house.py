import os
import numpy as np


DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'house')
DATASET_SIZE = 111
NUM_POINTS = 30
IMAGE_SHAPE = (384, 576)  # Height X Width (gray image, so channel is one)
POINTS_SHAPE = (NUM_POINTS, 2)


def get_image(index, data_dir=DATA_DIR):
  """
  Get the gray image
  """
  from PIL import Image
  assert 0 <= index <= DATASET_SIZE
  fpath = os.path.join(data_dir, 'images', 'house.seq{}.png'.format(index))
  image = np.asarray(Image.open(fpath))
  assert image.dtype == np.uint8
  assert image.shape == IMAGE_SHAPE
  return image


def get_points(index, dtype=None, data_dir=DATA_DIR):
  assert 0 <= index <= DATASET_SIZE
  fpath = os.path.join(data_dir, 'houses', 'house{}'.format(index + 1))
  points = np.loadtxt(fpath, dtype=dtype)
  if dtype is not None:
    assert points.dtype == dtype
  assert points.shape == POINTS_SHAPE
  return points


def get_all_images(data_dir=DATA_DIR):
  all_images = []
  for index in range(DATASET_SIZE):
    image = get_image(index, data_dir=data_dir)
    all_images.append(image)
  return all_images


def get_all_points(dtype=None, data_dir=DATA_DIR):
  all_points = []
  for index in range(DATASET_SIZE):
    points = get_points(index, dtype=dtype, data_dir=data_dir)
    all_points.append(points)
  return all_points


def show(ax, points, image=None):
  if image is None:
    H, W = IMAGE_SHAPE[:2]
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.invert_yaxis()
    ax.plot(points[:, 0], points[:, 1], 'o')
  else:
    ax.imshow(image, cmap='gray')
    ax.plot(points[:, 0], points[:, 1], 'o')
    ax.axis('off')


def main():
  import matplotlib.pyplot as plt

  fig, ax = plt.subplots()

  for index in range(DATASET_SIZE):
    points = get_points(index)
    image = get_image(index)

    assert points.shape == (30, 2)
    assert image.shape == (384, 576)

    print('{}: image:  {}, {}; points: {}, {}'.format(index, image.shape, image.dtype, points.dtype, points.shape))

    ax.cla()
    # show(ax, points)
    show(ax, points, image)
    plt.pause(0.1)


if __name__ == '__main__':
  main()
