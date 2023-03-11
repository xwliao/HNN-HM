# import numpy as np
# from functools import partial


def get_algorithms(dataset_name):
  algorithms = []

  # from methods.baseline import ground_truth_model  # noqa

  # algorithms.append({
  #     'name': 'Ground Truth',
  #     'model': ground_truth_model,
  #     'Color': 'k',
  #     'LineStyle': '--',
  #     'Marker': 'o'
  # })


  # from methods.baseline import random_model  # noqa

  # rng = np.random.default_rng(12345)
  # random_model_rng = partial(random_model, rng=rng)

  # algorithms.append({
  #     'name': 'Random Model',
  #     'model': random_model_rng,
  #     'Color': 'slategrey',
  #     'LineStyle': '--',
  #     'Marker': 'o'
  # })


  # from methods.baseline import dummy_model  # noqa

  # algorithms.append({
  #     'name': 'Dummy Model',
  #     'model': dummy_model,
  #     'Color': 'rosybrown',
  #     'LineStyle': '--',
  #     'Marker': 'o'
  # })


  # from methods.baseline import greedy_model  # noqa

  # algorithms.append({
  #     'name': 'Greedy Model',
  #     'model': greedy_model,
  #     'Color': 'indianred',
  #     'LineStyle': '--',
  #     'Marker': 'o'
  # })


  from methods.TM.tensor_matching import tensor_matching  # noqa

  algorithms.append({
      'name': 'TM',
      'model': tensor_matching,
      'Color': 'tab:pink',
      'LineStyle': '--',
      'Marker': 'o'
  })


  from methods.IPFP_HM.ipfp_hm import ipfp_hm  # noqa

  algorithms.append({
      'name': 'IPFP-HM',
      'model': ipfp_hm,
      'Color': 'indianred',
      'LineStyle': '-',
      'Marker': '^'
  })

  from methods.RRWHM.RRWHM import RRWHM  # noqa

  algorithms.append({
      'name': 'RRWHM',
      'model': RRWHM,
      'Color': 'tab:purple',
      'LineStyle': '--',
      'Marker': 'v'
  })


  from methods.BCAGM.bcagm import bcagm_mp  # noqa

  algorithms.append({
      'name': 'BCAGM',
      'model': bcagm_mp,
      'Color': 'tab:green',
      'LineStyle': '-',
      'Marker': 's',
      'directed': True
  })


  from methods.BCAGM.bcagm import adapt_bcagm3_mp  # noqa

  algorithms.append({
      'name': 'BCAGM3',
      'model': adapt_bcagm3_mp,
      'Color': 'darkcyan',
      'LineStyle': '--',
      'Marker': 'd',
      'directed': True
  })


  from methods.ADGM.adgm import adgm1  # noqa

  algorithms.append({
      'name': 'ADGM1',
      'model': adgm1,
      'Color': 'tomato',
      'LineStyle': '-',
      'Marker': 'p',
  })


  from methods.ADGM.adgm import adgm2  # noqa

  algorithms.append({
      'name': 'ADGM2',
      'model': adgm2,
      'Color': 'mediumvioletred',
      'LineStyle': '--',
      'Marker': '*',
  })


  from methods.HNN_HM.hnn_hm.evaluate import get_predict_function  # noqa

  algorithms.append({
      'name': "HNN-HM",
      'model': get_predict_function(dataset_name=dataset_name),
      # 'Color': 'turquoise',
      'Color': 'tab:blue',
      'LineStyle': '-',
      'Marker': 'H'
  })

  return algorithms
