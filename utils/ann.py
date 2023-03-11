from sklearn.neighbors import NearestNeighbors


def annquery(Xr, Xq, k):
    """
    Performs Approximate K-Nearest-Neighbor query for a set of points

    Usage:
       nnidx, dists = annquery(Xr, Xq, k)

    Arguments:
      - Xr: the reference points (n x d matrix)
      - Xq: the query points (nq x d matrix)
      - k:  the number of neighbors for each query point

    Outputs:
      - indices:   nq x k
      - distances: nq x k
   """
    nbrs = NearestNeighbors(n_neighbors=k).fit(Xr)
    # distances: nq * k
    # indices: nq * k
    distances, indices = nbrs.kneighbors(Xq)
    return indices, distances
