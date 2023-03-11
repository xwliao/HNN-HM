def get_config(dataset_name):
  if dataset_name == "Pascal VOC":
    from .pascal_voc import PascalVOCConfig
    cfg = PascalVOCConfig()
  elif dataset_name == "SPair-71k":
    from .spair import SpairConfig
    cfg = SpairConfig()
  elif dataset_name == "Willow":
    from .willow import WillowConfig
    cfg = WillowConfig()
  elif dataset_name == "House":
    from .house import HouseConfig
    cfg = HouseConfig()
  elif dataset_name == "Synthetic":
    from .synthetic import SyntheticConfig
    cfg = SyntheticConfig()

  assert cfg.DATASET_NAME == dataset_name

  return cfg
