from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import queue
import tensorflow as tf
import threading

SMALL_NUMBER = 1e-7


def glorot_init(shape):
    initialization_range = np.sqrt(6.0 / (shape[-2] + shape[-1]))
    return np.random.uniform(low=-initialization_range, high=initialization_range, size=shape).astype(np.float32)


class ThreadedIterator(object):
  def __init__(self, original_iterator, max_queue_size=2):
    self._queue = queue.Queue(maxsize=max_queue_size)
    self._thread = threading.Thread(target=lambda: self.worker(original_iterator))
    self._thread.start()

  def worker(self, original_iterator):
    for element in original_iterator:
      assert element is not None, 'By convention, iterator elements must not be None'
      self._queue.put(element, block=True)
    self._queue.put(None, block=True)

  def __iter__(self):
    next_element = self._queue.get(block=True)
    while next_element is not None:
      yield next_element
      next_element = self._queue.get(block=True)
    self._thread.join()


class MLP(object):
  def __init__(self, in_size, out_size, hidden_sizes, dropout_keep_prob):
    self._in_size = in_size
    self._out_size = out_size
    self._hidden_sizes = list(hidden_sizes)
    self._dropout_keep_prob = dropout_keep_prob
    self._params = self.make_network_params()

  def make_network_params(self):
    dims = [self._in_size] + self._hidden_sizes + [self._out_size]
    weight_sizes = list(zip(dims[:-1], dims[1:]))
    weights = [tf.Variable(self.init_weights(s), name='MLP_W_layer%i' % i)
               for (i, s) in enumerate(weight_sizes)]
    biases = [tf.Variable(np.zeros(s[-1]).astype(np.float32), name='MLP_b_layer%i' % i)
              for (i, s) in enumerate(weight_sizes)]
    network_params = {
        'weights': weights,
        'biases': biases,
    }
    return network_params

  def init_weights(self, shape):
    return glorot_init(shape)

  def __call__(self, inputs):
      acts = inputs
      for W, b in zip(self._params['weights'], self._params['biases']):
          hid = tf.matmul(acts, tf.nn.dropout(W, self._dropout_keep_prob)) + b
          acts = tf.nn.relu(hid)
      last_hidden = hid
      return last_hidden
