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

  n_cols = 3
  n_rows = (len(data) - 1) / n_cols + 1
  for i, d in enumerate(data):
    ax = plt.subplot(n_rows, n_cols, i + 1)
    adjacency = np.array(d['adjacency_matrix'])
    prediction = d['prediction']
    target = d['target']
    gr = nx.from_numpy_matrix(adjacency)
    nx.draw(gr, ax=ax, node_size=20)
    plt.title('%.2f (%.2f)' % (prediction[0], target[0]))
  plt.tight_layout()
  plt.show()


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Generates random graph')
  parser.add_argument('--evaluation_file', action='store', required=True, help='Path where the json file is stored')
  args = parser.parse_args()
  main(args)
