from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib.pylab as plt
import numpy as np

NODES = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16] # 20
EVAL_LOCAL =  [0.28770, 0.14329, 0.04665, 0.02415, 0.01424, 0.00386, 0.00752, 0.01625, 0.02802, 0.04138, 0.05448]  # 0.11457
EVAL_GLOBAL = [0.42419, 0.07106, 0.00638, 0.00213, 0.00261, 0.00367, 0.00583, 0.01072, 0.01734, 0.02665, 0.03792]  # 0.09479


def main():

  def outside(x):
    x = list(x)
    return x[:3] + x[-5:]

  def inside(x):
    x = list(x)
    return x[3:-5]

  width = 0.8

  plt.figure(figsize=(6, 4))

  # Local
  positions1 = range(len(NODES))
  plt.bar(outside(positions1), outside(EVAL_LOCAL), width, alpha=0.4, color='red')
  plt.bar(inside(positions1), inside(EVAL_LOCAL), width, alpha=1., color='red', label='Local (T=8)')

  positions2 = range(len(NODES) + 1, len(NODES) + len(NODES) + 1)
  plt.bar(outside(positions2), outside(EVAL_GLOBAL), width, alpha=0.4, color='blue')
  plt.bar(inside(positions2), inside(EVAL_GLOBAL), width, alpha=1., color='blue', label='Global (T=8)')

  plt.xticks(positions1 + positions2, NODES + NODES)

  plt.gca().yaxis.grid(True)
  plt.gca().set_yscale('log')
  plt.legend()
  plt.xlabel('L1 loss')
  plt.ylabel('Number of nodes')
  plt.show()

if __name__ == '__main__':
  main()
