import pickle
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from plot_utils import plot_results


def create_directory(save_dir):
  save_dir = Path(save_dir)
  if not save_dir.exists():
    save_dir.mkdir(parents=True)


def read_pickle(fpath):
  with open(fpath, 'rb') as handle:
      rets = pickle.load(handle)
  return rets


def write_pickle(fpath, results_dict):
  with open(fpath, 'wb') as handle:
      pickle.dump(results_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


def show(algorithms, xdata,
         mean_accuracy, mean_match_score, mean_time,
         xlabel=None, text_list=None,
         save_dir=None):
  """Plot results"""

  xdata = np.asarray(xdata)

  if save_dir is not None:
    save_dir = Path(save_dir)
    create_directory(save_dir)

  fig, ax = plt.subplots()
  plot_results(ax, algorithms,
               xdata, mean_accuracy,
               xlabel=xlabel,
               ylabel='Accuracy',
               text_list=text_list)
  if save_dir is not None:
    fig.savefig(save_dir / 'accuracy.pdf', bbox_inches='tight')

  fig, ax = plt.subplots()
  plot_results(ax, algorithms,
               xdata, mean_match_score,
               xlabel=xlabel,
               ylabel='Matching Score',
               text_list=text_list)
  if save_dir is not None:
    fig.savefig(save_dir / 'matching_score.pdf', bbox_inches='tight')

  fig, ax = plt.subplots()
  plot_results(ax, algorithms,
               xdata, mean_time,
               xlabel=xlabel,
               ylabel='Time',
               text_list=text_list)
  if save_dir is not None:
    fig.savefig(save_dir / 'time.pdf', bbox_inches='tight')

  if save_dir is None:
    plt.show()
