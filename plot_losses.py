from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import matplotlib.pylab as plt


def main(args):
  with open(args.log_file, 'r') as fp:
    data = json.load(fp)

  epochs = []
  train_abs = []
  valid_abs = []
  for d in data:
    epochs.append(d['epoch'])
    train_abs.append(d['train_results'][1][0])
    valid_abs.append(d['valid_results'][1][0])
  plt.plot(epochs, train_abs, label='Training')
  plt.plot(epochs, valid_abs, label='Evaluation')
  plt.xlabel('Epoch')
  plt.ylabel('Accucary')
  plt.legend()
  plt.show()


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Generates random graph')
  parser.add_argument('--log_file', action='store', required=True, help='Path where the json file is stored')
  args = parser.parse_args()
  main(args)
