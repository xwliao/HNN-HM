import os
import re
# from PIL import Image
# from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np


DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'PascalVOC')
ANNOTATION_DIR = os.path.join(DATA_DIR, 'annotations')
IMAGE_DIR = os.path.join(DATA_DIR, 'TrainVal', 'VOCdevkit', 'VOC2011', 'JPEGImages')

TRAIN_LIST_DIR = os.path.join(DATA_DIR, 'train')
TEST_LIST_DIR = os.path.join(DATA_DIR, 'test')

CATEGORIES = [
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor"
]

KEYPOINT_COORDINATES = ('x', 'y')  # Only (x, y), ignore the z coordinate
MIN_NUM_KEYPOINTS = 3

CROPPED_IMAGE_SHAPE = (256, 256, 3)  # Height X Width X Channel (RGB image)


def crop(image, keypoints, cropped_image_shape, bounds):
  """
  image: (H, W, C) array (can be None)
  keypoints: N x 2 array (can be None)
  """
  xmin, ymin, w, h = bounds[:]
  ph, pw = cropped_image_shape[:2]

  if image is not None:
    from PIL import Image
    img = Image.fromarray(image)
    img = img.resize((pw, ph), resample=Image.BICUBIC, box=(xmin, ymin, xmin + w, ymin + h))
    image = np.asarray(img)

  if keypoints is not None:
    assert keypoints.ndim == 2
    assert keypoints.shape[-1] == 2
    xscale = pw / w
    yscale = ph / h
    keypoints = (keypoints - [[xmin, ymin]]) * [[xscale, yscale]]

  return image, keypoints


def get_xml_root(xml_file):
  assert os.path.exists(xml_file), '{} does not exist.'.format(xml_file)
  with open(xml_file, 'r') as f:
      tree = ET.parse(f)
  root = tree.getroot()
  return root


def parse_category(root):
  category = root.find('./category').text
  return category


def parse_bounds(root):
  bounds = root.find('./visible_bounds').attrib
  h = float(bounds['height'])
  w = float(bounds['width'])
  xmin = float(bounds['xmin'])
  ymin = float(bounds['ymin'])
  return np.array([xmin, ymin, w, h])


def parse_keypoints(root, coordinates, dtype=np.float64):
  """
  coordinates: e.g. ('x', 'y', 'z' ) or ('x', 'y')

  Output:
    keypoints: N x d array, where d = len(attrib); N keypoints, each one is a d-dimensional vector
    keypoint_names: List of N string
  """
  keypoint_names = []
  keypoint_list = []
  for keypoint in root.findall('./keypoints/keypoint'):
    keypoint_names.append(keypoint.attrib['name'])
    keypoint_list.append([float(keypoint.attrib[k]) for k in coordinates])
  keypoints = np.asarray(keypoint_list, dtype=dtype)
  return keypoints, keypoint_names


def parse_image_name(root):
  image_name = root.find('./image').text + '.jpg'
  return image_name


def get_category(xml_relative_path):
  xml_file = os.path.join(ANNOTATION_DIR, xml_relative_path)
  root = get_xml_root(xml_file)
  return parse_category(root)


def get_bounds(xml_relative_path):
  xml_file = os.path.join(ANNOTATION_DIR, xml_relative_path)
  root = get_xml_root(xml_file)
  return parse_bounds(root)


def get_image_name(xml_relative_path):
  xml_file = os.path.join(ANNOTATION_DIR, xml_relative_path)
  root = get_xml_root(xml_file)
  return parse_image_name(root)


def get_image(xml_relative_path, cropped=False, dtype=np.uint8):
  """ Read Image in RGB format """
  from PIL import Image

  xml_file = os.path.join(ANNOTATION_DIR, xml_relative_path)
  root = get_xml_root(xml_file)

  image_name = parse_image_name(root)
  image_file = os.path.join(IMAGE_DIR, image_name)
  assert os.path.exists(image_file), '{} does not exist.'.format(image_file)
  image = np.asarray(Image.open(image_file), dtype=dtype)

  if cropped:
    bounds = parse_bounds(root)
    image, _ = crop(image=image, keypoints=None,
                    cropped_image_shape=CROPPED_IMAGE_SHAPE, bounds=bounds)
    assert image.shape == CROPPED_IMAGE_SHAPE

  assert image.dtype == dtype

  return image


def get_keypoints(xml_relative_path, cropped=False, dtype=np.float32):
  xml_file = os.path.join(ANNOTATION_DIR, xml_relative_path)
  root = get_xml_root(xml_file)

  keypoints, keypoint_names = parse_keypoints(root, coordinates=KEYPOINT_COORDINATES, dtype=dtype)

  if cropped and len(keypoints) > 0:
    bounds = parse_bounds(root)
    _, keypoints = crop(image=None, keypoints=keypoints,
                        cropped_image_shape=CROPPED_IMAGE_SHAPE, bounds=bounds)

  return keypoints, keypoint_names


def filter_xml_files(xml_list, min_num_keypoints):
  out_files = []
  for xml_file in xml_list:
    keypoints, _ = get_keypoints(xml_file)
    if len(keypoints) >= min_num_keypoints:
      out_files.append(xml_file)
  return out_files


def get_xml_files(list_file, min_num_keypoints=0):
  assert os.path.exists(list_file), '{} does not exist.'.format(list_file)
  with open(list_file) as f:
    xml_list = f.read().splitlines()  # `splitlines` can remove '\n'
  # Convert to right file separator
  xml_list = [re.sub(r'[\\\/]', os.sep, s) for s in xml_list]
  if min_num_keypoints > 0:
    # Remove files that contain keypoints less than `min_num_keypoints`
    xml_list = filter_xml_files(xml_list, min_num_keypoints=min_num_keypoints)
  return xml_list


def get_all_xml_files(list_files_dir, min_num_keypoints=0, categories=CATEGORIES):
  """
  Output:
  xmls: dict; key is the category name, value is a list of relative paths to xml files
  """
  xmls = {}
  for category in categories:
    list_file = os.path.join(list_files_dir, category + '.txt')
    xml_list = get_xml_files(list_file, min_num_keypoints=min_num_keypoints)
    xmls[category] = xml_list
  return xmls


def show(ax, points, image=None, bounds=None):
  if image is None:
    ax.invert_yaxis()
  else:
    ax.imshow(image)
    ax.axis('off')

  if bounds is not None:
    import matplotlib.patches as patches
    xmin, ymin, w, h = bounds
    # Create a Rectangle patch
    rect = patches.Rectangle((xmin, ymin), w, h, edgecolor='g', facecolor='none')
    # Add the patch to the Axes
    ax.add_patch(rect)

  if points.shape[0] > 1:
    ax.plot(points[:, 0], points[:, 1], 'ro')


def main():
  import matplotlib.pyplot as plt

  fig, ax = plt.subplots()

  list_file_dir = TRAIN_LIST_DIR
  # list_file_dir = TEST_LIST_DIR

  xmls = get_all_xml_files(list_file_dir, min_num_keypoints=0, categories=CATEGORIES)

  cropped = True
  # cropped = False
  for category in CATEGORIES:
    for xml_file in xmls[category]:
      # xml_file = 'aeroplane/2009_003219_1.xml'  # Note: wrong annotation
      # xml_file = 'aeroplane/2011_001476_4.xml'  # Note: zero keypoints

      image = get_image(xml_file, cropped=cropped)
      points, names = get_keypoints(xml_file, cropped=cropped)
      bounds = get_bounds(xml_file) if not cropped else None

      print('{}: image:  {}, {}; points: {}, {}'.format(xml_file, image.shape, image.dtype, points.dtype, points.shape))

      ax.cla()
      show(ax, points, image, bounds)
      plt.pause(0.1)


if __name__ == '__main__':
  main()
