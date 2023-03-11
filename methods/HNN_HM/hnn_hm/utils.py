import tensorflow as tf


def _segments_to_one(data, segment_lenghts, aggregate_fn=tf.math.segment_sum):
  segment_ids = tf.repeat(tf.range(tf.size(segment_lenghts)), repeats=segment_lenghts, axis=0)
  return aggregate_fn(data, segment_ids)


def _one_to_segments(data, segment_lenghts):
  return tf.repeat(data, repeats=segment_lenghts, axis=0)


def _unsorted_segments_to_one(data, segment_ids, num_segments, aggregate_fn=tf.math.unsorted_segment_sum):
  return aggregate_fn(data, segment_ids, num_segments)


def _one_to_unsorted_segments(data, segment_ids):
  return tf.gather(data, segment_ids)


# def _fixed_number_to_one(self, data, indices, aggregate_fn=tf.math.segment_sum):
#   """
#   indices: tensor of shape N x fixed_number
#   """
#   num_segments = tf.shape(indices)[0]
#   segment_ids = tf.broadcast_to(tf.reshape(tf.range(num_segments), [-1, 1]), tf.shape(indices))
#   segment_ids = tf.reshape(segment_ids, [-1])
#   indices = tf.reshape(indices, [-1])

#   data_expand = tf.gather(data, indices)
#   return aggregate_fn(data_expand, segment_ids=segment_ids)


def _fixed_number_to_one(data, indices, aggregate_fn=tf.math.reduce_sum):
  """
  indices: tensor of shape N x fixed_number
  """
  return aggregate_fn(tf.gather(data, indices), axis=1)


def _one_to_fixed_number(data, indices, n_out, aggregate_fn=tf.math.unsorted_segment_sum):
  """
  indices: tensor of shape N x fixed_number
  """
  num_segments = n_out
  segment_ids = tf.reshape(indices, [-1])
  indices = tf.broadcast_to(tf.reshape(tf.range(tf.shape(indices)[0]), [-1, 1]), tf.shape(indices))
  indices = tf.reshape(indices, [-1])

  data_expand = tf.gather(data, indices)
  return aggregate_fn(data_expand, segment_ids=segment_ids, num_segments=num_segments)


def _unsorted_segment_softmax(data,
                              segment_ids,
                              num_segments,
                              name="unsorted_segment_softmax"):
  """Performs an elementwise softmax operation along segments of a tensor.

  The input parameters are analogous to `tf.math.unsorted_segment_sum`. It
  produces an output of the same shape as the input data, after performing an
  elementwise sofmax operation between all of the rows with common segment id.

  Args:
    data: A tensor with at least one dimension.
    segment_ids: A tensor of indices segmenting `data` across the first
      dimension.
    num_segments: A scalar tensor indicating the number of segments. It should
      be at least `max(segment_ids) + 1`.
    name: A name for the operation (optional).

  Returns:
    A tensor with the same shape as `data` after applying the softmax operation.

  """
  with tf.name_scope(name):
    segment_maxes = tf.math.unsorted_segment_max(data, segment_ids,
                                                 num_segments)
    maxes = tf.gather(segment_maxes, segment_ids)
    # Possibly refactor to `tf.stop_gradient(maxes)` for better performance.
    data -= maxes
    exp_data = tf.exp(data)
    segment_sum_exp_data = tf.math.unsorted_segment_sum(exp_data, segment_ids,
                                                        num_segments)
    sum_exp_data = tf.gather(segment_sum_exp_data, segment_ids)
    return exp_data / sum_exp_data


def _one_to_fixed_number_attention(queries, keys, values,
                                   indices, aggregate_fn=tf.math.unsorted_segment_sum):
  """
  queries: #query x #head x key_size
  keys: #key x #head x key_size
  values: #key x #head x value_size
  indices: #key x m, each key corresponding to m queries

  output: #query x #head x value_size
  """
  num_segments = tf.shape(queries)[0]
  segment_ids = tf.reshape(indices, [-1])
  # size: #key x m
  indices = tf.broadcast_to(tf.reshape(tf.range(tf.shape(indices)[0]), [-1, 1]), tf.shape(indices))
  # size: (#key * m)
  indices = tf.reshape(indices, [-1])

  # size: (#key * m) x #head x key_size
  queries_expand = tf.gather(queries, segment_ids)
  # size: (#key * m) x #head x key_size
  keys_expand = tf.gather(keys, indices)

  # size: (#key * m) x #head
  attention_weights_logits = tf.reduce_sum(queries_expand * keys_expand, axis=-1)
  attention_weights = _unsorted_segment_softmax(attention_weights_logits,
                                                segment_ids=segment_ids,
                                                num_segments=num_segments)

  # size: (#key * m) x #head x value_size
  values_expand = tf.gather(values, indices)
  # size: (#key * m) x #head x value_size
  attented_values = values_expand * attention_weights[..., None]

  # size: #queries x #head x value_size
  return aggregate_fn(attented_values, segment_ids=segment_ids, num_segments=num_segments)


def _fixed_number_to_one_attention(queries, keys, values,
                                   indices, aggregate_fn=tf.math.segment_sum):
  """
  queries: #query x #head x key_size
  keys: #key x #head x key_size
  values: #key x #head x value_size
  indices: #query x m, each query corresponding to m keys

  output: #query x #head x value_size
  """
  num_segments = tf.shape(queries)[0]
  segment_ids = tf.broadcast_to(tf.reshape(tf.range(num_segments), [-1, 1]), tf.shape(indices))
  segment_ids = tf.reshape(segment_ids, [-1])

  indices_flat = tf.reshape(indices, [-1])

  # size: (#query * m) x #head x key_size
  queries_expand = tf.gather(queries, segment_ids)
  # size: (#query * m) x #head x key_size
  keys_expand = tf.gather(keys, indices_flat)

  # size: (#query * m) x #head
  attention_weights_logits = tf.reduce_sum(queries_expand * keys_expand, axis=-1)
  attention_weights = _unsorted_segment_softmax(attention_weights_logits,
                                                segment_ids=segment_ids,
                                                num_segments=num_segments)

  # size: (#query * m) x #head x value_size
  values_expand = tf.gather(values, indices_flat)
  # size: (#query * m) x #head x value_size
  attented_values = values_expand * attention_weights[..., None]

  # size: #queries x #head x value_size
  return aggregate_fn(attented_values, segment_ids=segment_ids)


# # TODO: bug? nan.
def unsorted_segment_normalization(data, segment_ids, num_segments, method):
  """
  `method`: should be one of ['l1', 'l2', 'softmax']
  """
  assert method in ['l1', 'l2', 'softmax']
  if method == 'l1':
    data2 = tf.abs(data)
    segment_sum = tf.math.unsorted_segment_sum(data2, segment_ids, num_segments)
    inv_segment_sum = tf.math.reciprocal_no_nan(segment_sum)
    scale = tf.gather(inv_segment_sum, segment_ids)
    scale = tf.reshape(scale, tf.shape(data))
    data_out = data * scale
  elif method == 'l2':
    data2 = tf.square(data)
    segment_sum = tf.math.unsorted_segment_sum(data2, segment_ids, num_segments)
    segment_sum = tf.sqrt(segment_sum)
    inv_segment_sum = tf.math.reciprocal_no_nan(segment_sum)
    scale = tf.gather(inv_segment_sum, segment_ids)
    scale = tf.reshape(scale, tf.shape(data))
    data_out = data * scale
  elif method == 'softmax':
    segment_max = tf.math.unsorted_segment_max(data, segment_ids, num_segments)
    data_max = tf.gather(segment_max, segment_ids)
    data_exp = tf.exp(data - data_max)
    segment_sum = tf.math.unsorted_segment_sum(data_exp, segment_ids, num_segments)
    inv_segment_sum = tf.math.reciprocal_no_nan(segment_sum)
    scale = tf.gather(inv_segment_sum, segment_ids)
    scale = tf.reshape(scale, tf.shape(data_exp))
    data_out = data_exp * scale

  return data_out


# def unsorted_segment_normalization(data, segment_ids, num_segments, method):
#   """
#   `method`: should be one of ['l1', 'l2', 'softmax']
#   """
#   assert method in ['l1', 'l2', 'softmax']
#   if method == 'l1':
#     epsilon = 1e-6
#     data2 = tf.abs(data)
#     segment_sum = tf.math.unsorted_segment_sum(data2, segment_ids, num_segments)
#     segment_sum = tf.maximum(segment_sum, epsilon)
#     inv_segment_sum = tf.math.reciprocal(segment_sum)
#     scale = tf.gather(inv_segment_sum, segment_ids)
#     scale = tf.reshape(scale, tf.shape(data))
#     data_out = tf.multiply(data, scale)
#   elif method == 'l2':
#     epsilon = 1e-12
#     data2 = tf.square(data)
#     segment_sum = tf.math.unsorted_segment_sum(data2, segment_ids, num_segments)
#     segment_sum = tf.maximum(segment_sum, epsilon)
#     inv_segment_sum = tf.math.rsqrt(segment_sum)
#     scale = tf.gather(inv_segment_sum, segment_ids)
#     scale = tf.reshape(scale, tf.shape(data))
#     data_out = tf.multiply(data, scale)
#   elif method == 'softmax':
#     segment_max = tf.math.unsorted_segment_max(data, segment_ids, num_segments)
#     data_max = tf.gather(segment_max, segment_ids)
#     data_exp = tf.exp(data - data_max)
#     segment_sum = tf.math.unsorted_segment_sum(data_exp, segment_ids, num_segments)
#     inv_segment_sum = tf.math.reciprocal(segment_sum)
#     scale = tf.gather(inv_segment_sum, segment_ids)
#     scale = tf.reshape(scale, tf.shape(data_exp))
#     data_out = tf.multiply(data, scale)

#   return data_out


# def unsorted_segment_normalization(data, segment_ids, num_segments, method):
#   """
#   `method`: should be one of ['l1', 'l2', 'softmax']
#   """
#   assert method in ['l1', 'l2', 'softmax']
#   if method == 'l1':
#     epsilon = 1e-6
#     data2 = tf.abs(data)
#     segment_sum = tf.math.unsorted_segment_sum(data2, segment_ids, num_segments)
#     inv_segment_sum = tf.where(segment_sum >= epsilon,
#                                tf.math.reciprocal(segment_sum),
#                                tf.zeros_like(segment_sum))
#     scale = tf.gather(inv_segment_sum, segment_ids)
#     scale = tf.reshape(scale, tf.shape(data))
#     data_out = tf.multiply(data, scale)
#   elif method == 'l2':
#     epsilon = 1e-12
#     data2 = tf.square(data)
#     segment_sum = tf.math.unsorted_segment_sum(data2, segment_ids, num_segments)
#     inv_segment_sum = tf.where(segment_sum >= epsilon,
#                                tf.math.rsqrt(segment_sum),
#                                tf.zeros_like(segment_sum))
#     scale = tf.gather(inv_segment_sum, segment_ids)
#     scale = tf.reshape(scale, tf.shape(data))
#     data_out = tf.multiply(data, scale)
#   elif method == 'softmax':
#     segment_max = tf.math.unsorted_segment_max(data, segment_ids, num_segments)
#     data_max = tf.gather(segment_max, segment_ids)
#     data_exp = tf.exp(data - data_max)
#     segment_sum = tf.math.unsorted_segment_sum(data_exp, segment_ids, num_segments)
#     inv_segment_sum = tf.math.reciprocal(segment_sum)
#     scale = tf.gather(inv_segment_sum, segment_ids)
#     scale = tf.reshape(scale, tf.shape(data_exp))
#     data_out = tf.multiply(data, scale)

#   return data_out


def normalize_rows(nodes, n_rows, row_id, method='l1'):
  """Normalize rows

  `n_rows` should be the total number of all rows.
  `method`: set `unsorted_segment_normalization`
  """
  nodes = unsorted_segment_normalization(nodes, segment_ids=row_id,
                                         num_segments=n_rows, method=method)
  return nodes


def normalize_cols(nodes, n_cols, col_id, method='l1'):
  """Normalize columns

  `n_cols` should be the total number of all columns.
  `method`: set `unsorted_segment_normalization`
  """
  nodes = unsorted_segment_normalization(nodes, segment_ids=col_id,
                                         num_segments=n_cols, method=method)
  return nodes


def normalize(nodes, rows, cols, row_id, col_id, max_iter, method='l1', row_is_last=False):
  """Normalize rows/cols
  `method`: set `unsorted_segment_normalization`
  """
  n_all_rows = tf.reduce_sum(rows)
  n_all_cols = tf.reduce_sum(cols)
  if not row_is_last:
    for _ in tf.range(max_iter):
      nodes = normalize_rows(nodes, n_rows=n_all_rows, row_id=row_id, method=method)
      nodes = normalize_cols(nodes, n_cols=n_all_cols, col_id=col_id, method=method)
  else:
    for _ in tf.range(max_iter):
      nodes = normalize_cols(nodes, n_cols=n_all_cols, col_id=col_id, method=method)
      nodes = normalize_rows(nodes, n_rows=n_all_rows, row_id=row_id, method=method)

  return nodes


def normalize_rows_for_graph(graph, method='l1'):
  """Normalize rows for `graph.nodes`

  `graph`: `graph.nodes.shape` should be `[N, 1]`
  `method`: set `unsorted_segment_normalization`
  """
  nodes = normalize_rows(tf.reshape(graph.nodes, [-1]),
                         n_rows=tf.reduce_sum(graph.n_row),
                         row_id=graph.row_id,
                         method=method)

  nodes = tf.reshape(nodes, tf.shape(graph.nodes))

  return graph.replace(nodes=nodes)


def normalize_cols_for_graph(graph, method='l1'):
  """Normalize columns for `graph.nodes`

  `graph`: `graph.nodes.shape` should be `[N, 1]`
  `method`: set `unsorted_segment_normalization`
  """
  nodes = normalize_cols(tf.reshape(graph.nodes, [-1]),
                         n_cols=tf.reduce_sum(graph.n_col),
                         col_id=graph.col_id,
                         method=method)

  nodes = tf.reshape(nodes, tf.shape(graph.nodes))

  return graph.replace(nodes=nodes)


def normalize_for_graph(graph, max_iter, method='l1', row_is_last=False):
  """Normalize rows/cols for `graph.nodes`

  `graph`: `graph.nodes.shape` should be `[N, 1]`
  `method`: set `unsorted_segment_normalization`
  """

  nodes = normalize(tf.reshape(graph.nodes, [-1]),
                    rows=graph.n_row,
                    cols=graph.n_col,
                    row_id=graph.row_id,
                    col_id=graph.col_id,
                    max_iter=max_iter,
                    method=method,
                    row_is_last=row_is_last)

  nodes = tf.reshape(nodes, tf.shape(graph.nodes))

  return graph.replace(nodes=nodes)
