"""
Modified from [PCA-GM](https://github.com/Thinklab-SJTU/PCA-GM)
"""

from scipy.optimize import linear_sum_assignment
import numpy as np


def hungarian(s: np.ndarray):
    """
    Solve optimal LAP permutation by hungarian algorithm.
    :param s: input 2d tensor
    :return: optimal permutation matrix
    """
    row, col = linear_sum_assignment(-s)

    perm_mat = np.zeros_like(s)
    perm_mat[row, col] = 1

    return perm_mat
