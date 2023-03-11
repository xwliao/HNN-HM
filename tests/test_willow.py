from utils import willow
from utils.willow import get_mat_files
from utils.willow import get_mat_files_train
from utils.willow import get_mat_files_test


def _test_get_mat_files_same_helper(func, categories):
  for category in categories:
    mat_files1 = func(category)
    mat_files2 = func(category)

    for f1, f2 in zip(mat_files1, mat_files2):
      assert f1 == f2


def test_get_mat_files_same():
  _test_get_mat_files_same_helper(get_mat_files, willow.CATEGORIES)
  _test_get_mat_files_same_helper(get_mat_files_train, willow.CATEGORIES)
  _test_get_mat_files_same_helper(get_mat_files_test, willow.CATEGORIES)


if __name__ == '__main__':
  test_get_mat_files_same()
