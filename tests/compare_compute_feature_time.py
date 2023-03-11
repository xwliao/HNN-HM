import random
import time

import numpy as np

from utils.compute_feature import compute_feature
from utils.compute_feature_np import compute_feature as compute_feature_np


def set_random_seed(seed):
  random.seed(seed)
  np.random.seed(seed)


if __name__ == '__main__':
  set_random_seed(seed=1024)

  nP1 = 20
  P1 = np.random.rand(nP1, 2)

  nP2 = 20
  P2 = np.random.rand(nP2, 2)

  nT1 = 2
  T1 = np.array([[0, 1, 2], [1, 2, 0]], dtype=np.int32)

  start_time = time.time()
  N = 1000
  for k in range(N):
    feat1, feat2 = compute_feature(P1, P2, T1)
  run_time = time.time() - start_time
  print('compute_feature: {} seconds'.format(run_time / N))

  start_time = time.time()
  N = 1000
  for k in range(N):
    feat1, feat2 = compute_feature_np(P1, P2, T1)
  run_time_np = time.time() - start_time
  print('compute_feature_np: {} seconds'.format(run_time_np / N))

  print('np / c++:', run_time_np / run_time)
