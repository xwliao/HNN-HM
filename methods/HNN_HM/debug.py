import sys

from hnn_hm.config.utils import get_config
from hnn_hm.evaluate import get_predict_function

if __name__ == '__main__':
  assert len(sys.argv) > 1, f"Usage: {sys.argv[0]} <dataset_name>"
  dataset_name = sys.argv[1]

  cfg = get_config(dataset_name=dataset_name)
  predict = get_predict_function(cfg=cfg)

  predict2 = get_predict_function(dataset_name=dataset_name)
