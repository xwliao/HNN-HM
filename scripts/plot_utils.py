import numpy as np


def plot_results(ax, algorithms, xdata, ydata, xlabel=None, ylabel=None, text_list=[]):
  # plot_settings = {
  #     "lineWidth": 3,    # Line width
  #     "markerSize": 10,  # Marker Size
  #     "fontSize": 2,     # Font Size
  #     "font": '\fontname{times new roman}'  # Font default
  # }
  lineWidth = 3    # Line width
  markerSize = 10  # Marker Size
  fontSize = 20     # Font Size
  # font = '\fontname{times new roman}'  # Font default

  for k, algorithm in enumerate(algorithms):
    ax.plot(xdata, ydata[:, k],
            linewidth=lineWidth,
            color=algorithm["Color"],
            linestyle=algorithm["LineStyle"],
            marker=algorithm["Marker"],
            markersize=markerSize,
            label=algorithm["name"])

  xmin = np.min(xdata)
  xmax = np.max(xdata)
  ymin = np.min(ydata)
  ymax = np.max(ydata)

  dy = 0.02 * (ymax - ymin)
  ax.axis([xmin, xmax, ymin - dy, ymax + dy])
  if xlabel is not None:
    ax.set_xlabel(xlabel, fontsize=fontSize)
  if ylabel is not None:
    ax.set_ylabel(ylabel, fontsize=fontSize)

  for k, text in enumerate(text_list):
    xpos = xmin + 0.1 * (xmax - xmin)
    ypos = ymin + 0.1 * (len(text_list) - k + 1) * (ymax - ymin)
    ax.text(xpos, ypos, text, fontsize=fontSize)

  # ax.legend()
  ax.legend(loc='lower left')

  # ax.grid()
