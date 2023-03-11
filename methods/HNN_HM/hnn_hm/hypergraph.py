"""
Modified from https://github.com/deepmind/graph_nets
"""

import collections

NODES = "nodes"
EDGES = "edges"
ROWS = "rows"
COLS = "cols"
GLOBALS = "globals"

HYPEREDGES = "hyperedges"
ROW_ID = "row_id"
COL_ID = "col_id"

N_NODE = "n_node"
N_EDGE = "n_edge"
N_ROW = "n_row"
N_COL = "n_col"
N_GLOBAL = "n_global"

HYPEREDGES_FEATURE_FIELDS = (EDGES, ROWS, COLS)
HYPEREDGES_INDEX_FIELDS = (HYPEREDGES, ROW_ID, COL_ID)
HYPEREDGES_NUMBER_FIELDS = (N_EDGE, N_ROW, N_COL)
HYPEREDGES_INDEX_OF_NODES = (True, False, False)

HYPERGRAPH_FEATURE_FIELDS = (NODES, EDGES, ROWS, COLS, GLOBALS)
HYPERGRAPH_INDEX_FIELDS = (HYPEREDGES, ROW_ID, COL_ID)
HYPERGRAPH_DATA_FIELDS = (NODES, EDGES, HYPEREDGES, ROWS, ROW_ID, COLS, COL_ID, GLOBALS)
HYPERGRAPH_NUMBER_FIELDS = (N_NODE, N_EDGE, N_ROW, N_COL, N_GLOBAL)
ALL_FIELDS = (NODES, N_NODE,
              EDGES, N_EDGE, HYPEREDGES,
              ROWS, N_ROW, ROW_ID,
              COLS, N_COL, COL_ID,
              GLOBALS, N_GLOBAL)


class HypergraphsTuple(collections.namedtuple("HypergraphsTuple", ALL_FIELDS)):
  def _validate_none_fields(self):
    """Asserts that the set of `None` fields in the instance is valid."""
    for k in HYPERGRAPH_NUMBER_FIELDS:
      if getattr(self, k) is None:
        raise ValueError("Field `{}` cannot be None".format(k))
    for feat_k, ind_k in zip(HYPEREDGES_FEATURE_FIELDS, HYPEREDGES_INDEX_FIELDS):
      if getattr(self, ind_k) is None and getattr(self, feat_k) is not None:
        raise ValueError("Field `{}` must be None as field `{}` is None".format(feat_k, ind_k))

  def __init__(self, *args, **kwargs):
    del args, kwargs
    # The fields of a `namedtuple` are filled in the `__new__` method.
    # `__init__` does not accept parameters.
    super(HypergraphsTuple, self).__init__()
    self._validate_none_fields()

  def replace(self, **kwargs):
    output = self._replace(**kwargs)
    output._validate_none_fields()  # pylint: disable=protected-access
    return output

  def map(self, field_fn, fields=HYPERGRAPH_FEATURE_FIELDS):
    """Applies `field_fn` to the fields `fields` of the instance.

    `field_fn` is applied exactly once per field in `fields`. The result must
    satisfy the `HypergraphsTuple` requirement w.r.t. `None` fields.

    Args:
      field_fn: A callable that take a single argument.
      fields: (iterable of `str`). An iterable of the fields to apply
        `field_fn` to.

    Returns:
      A copy of the instance, with the fields in `fields` replaced by the result
      of applying `field_fn` to them.
    """
    return self.replace(**{k: field_fn(getattr(self, k)) for k in fields})
