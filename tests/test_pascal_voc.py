from utils import pascal_voc
from utils.pascal_voc import get_all_xml_files


def _test_get_all_xml_files_same_helper(list_files_dir, min_num_keypoints, categories):
  xml_files_all1 = get_all_xml_files(list_files_dir=list_files_dir,
                                     min_num_keypoints=min_num_keypoints,
                                     categories=categories)

  xml_files_all2 = get_all_xml_files(list_files_dir=list_files_dir,
                                     min_num_keypoints=min_num_keypoints,
                                     categories=categories)

  for category in categories:
    assert category in xml_files_all1.keys()
    assert category in xml_files_all2.keys()
    for f1, f2 in zip(xml_files_all1[category], xml_files_all2[category]):
      assert f1 == f2


def test_get_all_xml_files_same():
  _test_get_all_xml_files_same_helper(list_files_dir=pascal_voc.TRAIN_LIST_DIR,
                                      min_num_keypoints=0,
                                      categories=pascal_voc.CATEGORIES)

  _test_get_all_xml_files_same_helper(list_files_dir=pascal_voc.TRAIN_LIST_DIR,
                                      min_num_keypoints=3,
                                      categories=pascal_voc.CATEGORIES)

  _test_get_all_xml_files_same_helper(list_files_dir=pascal_voc.TEST_LIST_DIR,
                                      min_num_keypoints=0,
                                      categories=pascal_voc.CATEGORIES)

  _test_get_all_xml_files_same_helper(list_files_dir=pascal_voc.TEST_LIST_DIR,
                                      min_num_keypoints=3,
                                      categories=pascal_voc.CATEGORIES)


if __name__ == '__main__':
  test_get_all_xml_files_same()
