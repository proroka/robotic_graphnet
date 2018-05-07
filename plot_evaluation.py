from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import matplotlib.pylab as plt
import networkx as nx
import numpy as np


def main(args):
  with open(args.evaluation_file, 'r') as fp:
    data = json.load(fp)

  n_cols = 6
  n_rows = (len(data) - 1) / n_cols + 1
  for i, d in enumerate(data):
    ax = plt.subplot(n_rows, n_cols, i + 1)
    adjacency = np.array(d['adjacency_matrix'])
    prediction = np.array(d['prediction'])
    if len(prediction.shape) == 2:
      # Each node has a prediction.
      prediction = prediction[0][:adjacency.shape[0]]
      prediction_per_node = dict((k, '%.2f' % v) for k, v in enumerate(prediction))
      prediction = np.mean(prediction)
    else:
      prediction = prediction.item()
      prediction_per_node = None
    target = d['target'][0]
    gr = nx.from_numpy_matrix(adjacency)
    nx.draw(gr, ax=ax, node_size=20, with_labels=True, labels=prediction_per_node)
    plt.title('%.2f (%.2f)' % (prediction, target))
  plt.tight_layout()
  plt.show()

  # plt.figure(figsize=(6, 6))
  # d = data[11]
  # adjacency = np.array(d['adjacency_matrix'])
  # prediction = np.array(d['prediction'])
  # if len(prediction.shape) == 2:
  #   # Each node has a prediction.
  #   prediction = prediction[0][:adjacency.shape[0]]
  #   prediction_per_node = dict((k, '%.2f' % v) for k, v in enumerate(prediction))
  #   prediction = np.mean(prediction)
  # else:
  #   prediction = prediction.item()
  #   prediction_per_node = None
  # target = d['target'][0]
  # gr = nx.from_numpy_matrix(adjacency)
  # nx.set_edge_attributes(gr, 'length', .01)
  # nx.draw(gr, ax=plt.gca(), with_labels=True, labels=prediction_per_node,
  #         node_color='b',
  #         node_size=1000,
  #         alpha=0.8,
  #         linewidths=1)
  # plt.title('%.2f (%.2f)' % (prediction, target))
  # plt.show()

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Generates random graph')
  parser.add_argument('--evaluation_file', action='store', required=True, help='Path where the json file is stored')
  args = parser.parse_args()
  main(args)
