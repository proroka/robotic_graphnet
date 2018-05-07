from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import matplotlib.pylab as plt
import os


def main(args):
  for filename in args.log_file.split(','):
    with open(filename, 'r') as fp:
      data = json.load(fp)
    label = os.path.basename(filename)
    epochs = []
    train_abs = []
    valid_abs = []
    for d in data:
      epochs.append(d['epoch'])
      train_abs.append(d['train_results'][1][0])
      valid_abs.append(d['valid_results'][1][0])
    if args.plot_train:
      plt.plot(epochs, train_abs, label=label + ' (train)')
    plt.plot(epochs, valid_abs, label=label)

  plt.xlabel('Epoch')
  plt.ylabel('L1 loss')
  plt.legend()
  plt.show()


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Generates random graph')
  parser.add_argument('--log_file', action='store', required=True, help='Path where the json file is stored')
  parser.add_argument('--plot_train', action='store_true', help='Plots the training loss too')
  args = parser.parse_args()
  main(args)
