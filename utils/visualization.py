from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

import numpy as np

# import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from matplotlib.collections import LineCollection


class Hypergraph(
    collections.namedtuple("Hypergraph",
                           ("nodes", "edges", "hyperedges",
                            "n_node", "n_edge"))):
  def _validate_none_fields(self):
    """Asserts that the set of `None` fields in the instance is valid."""
    if self.n_node is None:
      raise ValueError("Field `n_node` cannot be None")
    if self.n_edge is None:
      raise ValueError("Field `n_edge` cannot be None")
    if self.hyperedges is None and self.edges is not None:
      raise ValueError("Field `edges` must be None as field `hyperedges` is None")

  def __init__(self, *args, **kwargs):
    del args, kwargs
    # The fields of a `namedtuple` are filled in the `__new__` method.
    # `__init__` does not accept parameters.
    super(Hypergraph, self).__init__()
    self._validate_none_fields()

  def replace(self, **kwargs):
    output = self._replace(**kwargs)
    output._validate_none_fields()  # pylint: disable=protected-access
    return output


def draw_hyperedges(ax, x, y, hyperedges, color=None, alpha=0.05, linewidth=2, linestyle='-', zorder=0):
  XY = np.column_stack([x, y])

  patches = []
  for hyperedge in hyperedges:
    xy = np.take(XY, hyperedge, axis=0)
    xy = np.reshape(xy, (-1, 2))
    closed = len(xy) > 2
    polygon = Polygon(xy, closed=closed)
    patches.append(polygon)

  p = PatchCollection(patches)

  p.set_facecolor('none')
  if color is not None:
    p.set_edgecolor(color)
  if alpha is not None:
    p.set_alpha(alpha)
  if linewidth is not None:
    p.set_linewidth(linewidth)
  if linestyle is not None:
    p.set_linestyle(linestyle)
  if zorder is not None:
    p.set_zorder(zorder)

  ax.add_collection(p)


def plot_hyper_graph(ax, graph, offset, color, label, s=120, **kwargs):
  offset = np.asarray(offset)
  data = graph.nodes + offset
  x = data[:, 0]
  y = data[:, 1]
  hyperedges = graph.hyperedges
  if hyperedges is not None:
    draw_hyperedges(ax, x, y, hyperedges=hyperedges, color=color)
  ax.scatter(x, y, c=color, edgecolors='none', s=s, label=label, **kwargs)
  return x, y


def draw_lines(ax, x1, y1, x2, y2, selected=None, color=None, label=None, alpha=0.8, linewidths=3, zorder=1):
  edges = np.column_stack([x1, y1, x2, y2])
  edges = edges[selected]

  segs = np.reshape(edges, (-1, 2, 2))

  line_segments = LineCollection(segs, linewidths=linewidths, zorder=zorder)

  if color is not None:
    line_segments.set_color(color)
  if alpha is not None:
    line_segments.set_alpha(alpha)
  if label is not None:
    line_segments.set_label(label)

  ax.add_collection(line_segments)


def plot_results(ax,
                 graph1, graph2,
                 assignmatrix, target,
                 title=None,
                 graph1_offset=(0, 0), graph2_offset=(0, 0),
                 graph1_color='C9', graph2_color='C1',
                 correct_color='C2', wrong_color='C3'):
  """
  assignmatrix: shape is (nP1, nP2)
  """

  plot_kwargs = {'alpha': 0.7, 'zorder': 2, 's': 120}
  x1, y1 = plot_hyper_graph(ax, graph1, offset=graph1_offset, color=graph1_color, label='graph1', **plot_kwargs)
  x2, y2 = plot_hyper_graph(ax, graph2, offset=graph2_offset, color=graph2_color, label='graph2', **plot_kwargs)

  n1 = len(x1)
  n2 = len(x2)

  x1 = np.repeat(x1, repeats=n2, axis=0)
  y1 = np.repeat(y1, repeats=n2, axis=0)

  x2 = np.tile(x2, n1)
  y2 = np.tile(y2, n1)

  assigned = assignmatrix.flatten() > 0

  line_plot_kwargs = {'alpha': 0.5, 'linewidths': 2, 'zorder': 1}

  corrected = target.flatten() > 0
  is_correct = np.logical_and(assigned, corrected)
  draw_lines(ax, x1, y1, x2, y2, selected=is_correct,
             color=correct_color, label='correct',
             **line_plot_kwargs)

  wronged = np.logical_not(corrected)
  is_wrong = np.logical_and(assigned, wronged)
  draw_lines(ax, x1, y1, x2, y2, selected=is_wrong,
             color=wrong_color, label='wrong',
             **line_plot_kwargs)

  plot_kwargs['alpha'] = 1
  plot_kwargs['c'] = 'none'
  plot_kwargs['linewidths'] = line_plot_kwargs['linewidths']
  # plot_kwargs['zorder'] = line_plot_kwargs['zorder']

  # Circle corresponding nodes
  # Use graph2's color to circle nodes of graph1
  ax.scatter(x1[corrected], y1[corrected], edgecolors=graph2_color, **plot_kwargs)
  # Use graph1's color to circle nodes of graph2
  ax.scatter(x2[corrected], y2[corrected], edgecolors=graph1_color, **plot_kwargs)

  # Mark correct pair of nodes
  ax.scatter(x1[is_correct], y1[is_correct], edgecolors=correct_color, **plot_kwargs)
  ax.scatter(x2[is_correct], y2[is_correct], edgecolors=correct_color, **plot_kwargs)

  # Mark wrong pair of nodes
  ax.scatter(x1[is_wrong], y1[is_wrong], edgecolors=wrong_color, **plot_kwargs)
  ax.scatter(x2[is_wrong], y2[is_wrong], edgecolors=wrong_color, **plot_kwargs)

  if title is not None:
    ax.set_title(title)

  ax.legend()
  ax.set_aspect('equal', adjustable='box')


def show(graph1, graph2, prediction, target,
         graph1_offset=(0, 0), graph2_offset=(0, 0),
         title=None, show=True, save_filename=None,
         figsize=(12.8, 9.6)):

  fig, ax = plt.subplots(figsize=figsize)
  plot_results(ax, graph1, graph2,
               assignmatrix=prediction,
               target=target,
               title=title,
               graph1_offset=graph1_offset,
               graph2_offset=graph2_offset)

  # plt.tight_layout()

  if save_filename is not None:
    fig.savefig(save_filename, bbox_inches='tight')

  if show:
    plt.show()
