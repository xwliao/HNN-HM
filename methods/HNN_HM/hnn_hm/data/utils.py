from .data import GraphCreator
from .data import Dataset


def get_problem_generator(cfg, rng=None):
  dataset_name = cfg.DATASET_NAME

  if dataset_name == "Pascal VOC":
    from .pascal_voc import PascalVOCProblemGenerator
    problem_generator = PascalVOCProblemGenerator(rng=rng)
  elif dataset_name == "SPair-71k":
    from .spair import SpairProblemGenerator
    problem_generator = SpairProblemGenerator(rng=rng)
  elif dataset_name == "Willow":
    from .willow import WillowProblemGenerator
    problem_generator = WillowProblemGenerator(rng=rng)
  elif dataset_name == "House":
    from .house import HouseProblemGenerator
    problem_generator = HouseProblemGenerator(rng=rng)
  elif dataset_name == "Synthetic":
    from .synthetic import SyntheticProblemGenerator
    problem_generator = SyntheticProblemGenerator(rng=rng)

  return problem_generator


def get_graph_creator(cfg):
    return GraphCreator(cfg)


def get_dataset(sets, cfg, rng=None):
  assert sets in ["train", "val", "test"]

  problem_generator = get_problem_generator(cfg, rng=rng)
  graph_creator = get_graph_creator(cfg)

  dataset = Dataset(problem_generator=problem_generator,
                    graph_creator=graph_creator)

  return dataset
