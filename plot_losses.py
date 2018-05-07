from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import matplotlib.pylab as plt
import numpy as np
import os


def smooth_plot(x, y, window=5, stride=1):
  def rolling_window(a, window):
    a = np.array(a)
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)
  return (np.mean(rolling_window(x, window), axis=-1)[::stride],
          np.median(rolling_window(y, window), axis=-1)[::stride],
          np.std(rolling_window(y, window), axis=-1)[::stride])


def plot(x, y, color, alpha, linestyle, label):
  x, y, sy = smooth_plot(x, y)
  plt.plot(x, y, color=color, linestyle=linestyle, lw=2, label=label, alpha=alpha)
  # plt.fill_between(x, y + sy, y - sy, facecolor=color, alpha=0.5)


def main(args):
  plt.figure(figsize=(6, 4))
  colors = ['r', 'g', 'b']
  linestyle = ['-', '--']
  alpha = [1., .4]

  for i, filename in enumerate(args.log_file.split(',')):
    color_mod = i % len(colors)
    style_mod = (i // len(colors)) % len(linestyle)

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
      plot(epochs, train_abs, alpha=alpha[style_mod], linestyle=':', color=colors[color_mod], label=label + ' (train)')
    plot(epochs, valid_abs, alpha=alpha[style_mod], linestyle=linestyle[style_mod], color=colors[color_mod], label=label)

  plt.xlabel('Epoch')
  plt.ylabel('L1 loss')
  plt.ylim(bottom=0, top=.1)
  plt.gca().yaxis.grid(True)
  plt.legend()
  plt.show()


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Generates random graph')
  parser.add_argument('--log_file', action='store', required=True, help='Path where the json file is stored')
  parser.add_argument('--plot_train', action='store_true', help='Plots the training loss too')
  args = parser.parse_args()
  main(args)
