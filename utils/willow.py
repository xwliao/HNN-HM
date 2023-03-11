import os
import numpy as np
import scipy.io as sio
from pathlib import Path


DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'WILLOW-ObjectClass_dataset', 'WILLOW-ObjectClass')

NUM_POINTS = 10
POINTS_SHAPE = (NUM_POINTS, 2)

# CROPPED_IMAGE_SHAPE = (256, 256, 3)  # Height X Width X Channel (RGB image)

CATEGORIES = [
    "Car",
    "Duck",
    "Face",
    "Motorbike",
    "Winebottle"
]

NUM_SAMPLES = {
    "Car": 40,
    "Duck": 50,
    "Face": 109 - 1,  # Ignore 'Face/image_0160.mat'
    "Motorbike": 40,
    "Winebottle": 66
}

NUM_SAMPLES_TRAIN = 20


def get_category(mat_file):
  # TODO: Use more robust to get category name
  return Path(mat_file).parent.name


def _get_image_file(mat_file):
  """ Replace `file.mat` to `file.png` """
  return mat_file[:-3] + 'png'


def detect_keypoints(image, method='SIFT'):
  """
  Input:
    image: RGB image of shape (H, W, 3) or gray image of shape (H, w)
    method: 'SIFT' or 'MSER'

  Output:
    points: (num_points, 2)
  """
  import cv2 as cv
  assert method in ['SIFT', 'MSER']
  if method == 'SIFT':
    # TODO: use parameters as BCAGM
    detector = cv.SIFT_create()
    # detector = cv.SIFT_create(contrastThreshold=0.01)
    # detector = cv.SIFT_create(contrastThreshold=0.03)
  else:
    # TODO: use parameters as BCAGM
    # 'MinDiversity', 0.4, 'MaxVariation', 0.3, 'Delta',10
    detector = cv.MSER_create()
    # detector = cv.MSER_create(_min_diversity=0.4, _max_variation=0.3, _delta=10)
  if image.ndim == 2:
    gray = image
  else:
    gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
  keypoints = detector.detect(gray)
  points = cv.KeyPoint_convert(keypoints)
  return points


def get_image(mat_file, dtype=np.uint8):
  """ Get Image in RGB format """
  from PIL import Image
  image_file = _get_image_file(mat_file)
  assert os.path.exists(image_file), '{} does not exist.'.format(image_file)
  image = np.asarray(Image.open(image_file), dtype=dtype)
  return image


def get_points(mat_file, dtype=None):
  """
  Output:
    points: shape is (num_points, 2),
            each row is a coordinate (x, y) in the image plane
  """
  with open(mat_file, 'rb') as f:
    data = sio.loadmat(f)
    # shape of points: (2, num_points)
    points = np.asarray(data['pts_coord'], dtype=dtype)
    # transpose to (num_points, 2)
    points = np.transpose(points)
  if dtype is not None:
    assert points.dtype == dtype
  assert points.shape == POINTS_SHAPE
  return points


def generate_points(mat_file, npts, replace=False, method='SIFT', rng=None):
  """ Generate keypoints randomly """
  rng = np.random.default_rng(rng)
  image = get_image(mat_file)
  points = detect_keypoints(image=image, method=method)
  inds = rng.choice(np.arange(len(points)), size=npts, replace=replace)
  return points[inds]


def get_mat_files(category, data_dir=DATA_DIR):
  assert category in CATEGORIES
  mat_dir = (Path(data_dir) / category)
  if category == 'Face':
    mat_files = [str(p) for p in mat_dir.glob('*.mat') if p.name != 'image_0160.mat']
  else:
    mat_files = [str(p) for p in mat_dir.glob('*.mat')]
  mat_files.sort()
  assert len(mat_files) == NUM_SAMPLES[category]
  return mat_files


def get_mat_files_train(category, data_dir=DATA_DIR):
  mat_files = get_mat_files(category, data_dir=DATA_DIR)
  return mat_files[:NUM_SAMPLES_TRAIN]


def get_mat_files_test(category, data_dir=DATA_DIR):
  mat_files = get_mat_files(category, data_dir=DATA_DIR)
  return mat_files[NUM_SAMPLES_TRAIN:]


def show(ax, points, image=None):
  if image is None:
    ax.invert_yaxis()
    ax.plot(points[:, 0], points[:, 1], 'go')
  else:
    ax.imshow(image, cmap='gray')
    ax.plot(points[:, 0], points[:, 1], 'go')
    ax.axis('off')


def main():
  import matplotlib.pyplot as plt

  fig, ax = plt.subplots()

  for category in CATEGORIES:
    mat_files = get_mat_files(category)
    for mat_file in mat_files:
      points = get_points(mat_file)
      image = get_image(mat_file)

      print('{}: image:  {}, {}; points: {}, {}'.format(mat_file, image.shape, image.dtype, points.dtype, points.shape))

      ax.cla()
      # show(ax, points)
      show(ax, points, image)
      # plt.pause(0.1)
      plt.pause(0.5)


if __name__ == '__main__':
  main()
