from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import collections
import json
import numpy as np
import tensorflow as tf

import graphnet_base
import utils


def _compute_adjacency_matrix(graph, max_n_vertices, num_edge_types, tie_fwd_bkwd=True):
  bwd_edge_offset = 0 if tie_fwd_bkwd else (num_edge_types // 2)
  amat = np.zeros((num_edge_types, max_n_vertices, max_n_vertices))
  for src, e, dest in graph:
    amat[e-1, dest, src] = 1
    amat[e-1 + bwd_edge_offset, src, dest] = 1
  return amat


class DenseGraphNet(graphnet_base.GraphNetBase):
  def __init__(self, args):
    super(DenseGraphNet, self).__init__(args)

  @classmethod
  def add_arguments(cls, parser):
    super(DenseGraphNet, cls).add_arguments(parser)
    parser.add_argument('--batch_size', type=int, action='store', default=256, help='Mini-batch size.')

  def set_arguments(self, args, params):
    params['batch_size'] = args.batch_size

  @classmethod
  def default_params(cls):
    params = dict(super(DenseGraphNet, cls).default_params())
    params.update({
        'batch_size': 256,
        'graph_state_dropout_keep_prob': 1.,
        'use_edge_bias': True,
    })
    return params

  def prepare_model(self):
    h_dim = self.parameters['hidden_size']
    # inputs
    self.placeholders['graph_state_keep_prob'] = tf.placeholder(tf.float32, None, name='graph_state_keep_prob')
    self.placeholders['initial_node_representation'] = tf.placeholder(tf.float32, [None, None, self.parameters['hidden_size']], name='node_features')
    self.placeholders['node_mask'] = tf.placeholder(tf.float32, [None, None], name='node_mask')
    self.placeholders['num_vertices'] = tf.placeholder(tf.int32, ())
    self.placeholders['adjacency_matrix'] = tf.placeholder(tf.float32, [None, self.num_edge_types, None, None])  # [b, e, v, v]
    self._adjacency_matrix = tf.transpose(self.placeholders['adjacency_matrix'], [1, 0, 2, 3])                   # [e, b, v, v]
    # weights
    self.weights['edge_weights'] = tf.Variable(utils.glorot_init([self.num_edge_types, h_dim, h_dim]))
    if self.parameters['use_edge_bias']:
      self.weights['edge_biases'] = tf.Variable(np.zeros([self.num_edge_types, 1, h_dim]).astype(np.float32))
    with tf.variable_scope('gru_scope'):
      cell = tf.contrib.rnn.GRUCell(h_dim)
      cell = tf.nn.rnn_cell.DropoutWrapper(cell, state_keep_prob=self.placeholders['graph_state_keep_prob'])
      self.weights['node_gru'] = cell

  def compute_final_node_representations(self):
    v = self.placeholders['num_vertices']
    h_dim = self.parameters['hidden_size']
    h = self.placeholders['initial_node_representation']                            # [b, v, h]
    h = tf.reshape(h, [-1, h_dim])

    # Precompute edge biases.
    if self.parameters['use_edge_bias']:
      biases = []                                                                   # e ; t ; [b*v, h]
      for edge_type, a in enumerate(tf.unstack(self._adjacency_matrix, axis=0)):
        summed_a = tf.reshape(tf.reduce_sum(a, axis=-1), [-1, 1])                   # [b*v, 1]
        biases.append(tf.matmul(summed_a, self.weights['edge_biases'][edge_type]))  # [b*v, h]

    with tf.variable_scope('gru_scope'):
      for i in range(self.parameters['num_timesteps']):
        if i > 0:
          tf.get_variable_scope().reuse_variables()
        for edge_type in range(self.num_edge_types):
          m = tf.matmul(h, self.weights['edge_weights'][edge_type])  # [b*v, h]
          if self.parameters['use_edge_bias']:
            m += biases[edge_type]                                   # [b*v, h]
          m = tf.reshape(m, [-1, v, h_dim])                          # [b, v, h]
          if edge_type == 0:
            acts = tf.matmul(self._adjacency_matrix[edge_type], m)
          else:
            acts += tf.matmul(self._adjacency_matrix[edge_type], m)
        acts = tf.reshape(acts, [-1, h_dim])                         # [b*v, h]
        h = self.weights['node_gru'](acts, h)[1]                     # [b*v, h]
      last_h = tf.reshape(h, [-1, v, h_dim])
    return last_h

  def local_regression(self, last_h, regression_transform):
    v = self.placeholders['num_vertices']
    last_h = tf.reshape(last_h, [-1, self.parameters['hidden_size']])          # [b*v, h]
    return tf.reshape(regression_transform(last_h), [-1, v])                   # [b, v]

  def global_regression(self, last_h, regression_gate, regression_transform):
    # last_h: [b x v x h]
    gate_input = tf.concat([last_h, self.placeholders['initial_node_representation']], axis=2)  # [b, v, 2h]
    gate_input = tf.reshape(gate_input, [-1, 2 * self.parameters['hidden_size']])               # [b*v, 2h]
    last_h = tf.reshape(last_h, [-1, self.parameters['hidden_size']])                           # [b*v, h]
    gated_outputs = tf.nn.sigmoid(regression_gate(gate_input)) * regression_transform(last_h)   # [b*v, 1]
    gated_outputs = tf.reshape(gated_outputs, [-1, self.placeholders['num_vertices']])          # [b, v]
    masked_gated_outputs = gated_outputs * self.placeholders['node_mask']                       # [b x v]
    output = tf.reduce_sum(masked_gated_outputs, axis=1)                                        # [b]
    return output

  def process_raw_graphs(self, raw_data, is_training_data, bucket_sizes=None):
    # Batches graphs with the same number of nodes together.
    if bucket_sizes is None:
      # Having buckets seems to break things.
      # bucket_sizes = np.array(list(range(4, self.max_num_vertices + 1, 2)) + [self.max_num_vertices + 1])
      bucket_sizes = np.array([self.max_num_vertices])

    bucketed = collections.defaultdict(list)
    x_dim = len(raw_data[0]['node_features'][0])
    for d in raw_data:
      num_vertices = 1 + max(v for e in d['graph'] for v in [e[0], e[2]])
      chosen_bucket_idx = np.argmax(bucket_sizes > num_vertices)
      chosen_bucket_size = bucket_sizes[chosen_bucket_idx]
      n_active_nodes = len(d['node_features'])
      bucketed[chosen_bucket_idx].append({
          'adj_mat': _compute_adjacency_matrix(d['graph'], chosen_bucket_size, self.num_edge_types, self.parameters['tie_fwd_bkwd']),
          'init': d['node_features'] + [[0 for _ in range(x_dim)] for _ in range(chosen_bucket_size - n_active_nodes)],
          'labels': [d['targets'][task_id][0] for task_id in self.parameters['task_ids']],
          'mask': [1. for _ in range(n_active_nodes)] + [0. for _ in range(chosen_bucket_size - n_active_nodes)]
      })

    if is_training_data:
      for bucket_idx, bucket in bucketed.items():
        np.random.shuffle(bucket)
        for task_id in self.parameters['task_ids']:
          task_sample_ratio = self.parameters['task_sample_ratios'].get(str(task_id))
          if task_sample_ratio is not None:
            ex_to_sample = int(len(bucket) * task_sample_ratio)
            for ex_id in range(ex_to_sample, len(bucket)):
              bucket[ex_id]['labels'][task_id] = None

    bucket_at_step = [[bucket_idx for _ in range(len(bucket_data) // self.parameters['batch_size'])]
                      for bucket_idx, bucket_data in bucketed.items()]
    bucket_at_step = [x for y in bucket_at_step for x in y]
    return bucketed, bucket_sizes, bucket_at_step

  def _pad_annotations(self, annotations):
    return np.pad(annotations,
                  pad_width=[[0, 0], [0, 0], [0, self.parameters['hidden_size'] - self.annotation_size]],
                  mode='constant')

  def make_batch(self, elements):
    batch_data = {'adj_mat': [], 'init': [], 'labels': [], 'node_mask': [], 'task_masks': []}
    for d in elements:
      batch_data['adj_mat'].append(d['adj_mat'])
      batch_data['init'].append(d['init'])
      batch_data['node_mask'].append(d['mask'])

      target_task_values = []
      target_task_mask = []
      for target_val in d['labels']:
        if target_val is None:
          target_task_values.append(0.)
          target_task_mask.append(0.)
        else:
          target_task_values.append(target_val)
          target_task_mask.append(1.)
      batch_data['labels'].append(target_task_values)
      batch_data['task_masks'].append(target_task_mask)
    return batch_data

  def make_minibatch_iterator(self, data, is_training):
    (bucketed, bucket_sizes, bucket_at_step) = data
    if is_training:
      np.random.shuffle(bucket_at_step)
      for _, bucketed_data in bucketed.items():
        np.random.shuffle(bucketed_data)

    bucket_counters = collections.defaultdict(int)
    dropout_keep_prob = self.parameters['graph_state_dropout_keep_prob'] if is_training else 1.
    for step in range(len(bucket_at_step)):
      bucket = bucket_at_step[step]
      start_idx = bucket_counters[bucket] * self.parameters['batch_size']
      end_idx = (bucket_counters[bucket] + 1) * self.parameters['batch_size']
      elements = bucketed[bucket][start_idx:end_idx]
      batch_data = self.make_batch(elements)

      num_graphs = len(batch_data['init'])
      initial_representations = batch_data['init']
      initial_representations = self._pad_annotations(initial_representations)

      batch_feed_dict = {
          self.placeholders['initial_node_representation']: initial_representations,
          self.placeholders['target_values']: np.transpose(batch_data['labels'], axes=[1, 0]),
          self.placeholders['target_mask']: np.transpose(batch_data['task_masks'], axes=[1, 0]),
          self.placeholders['num_graphs']: num_graphs,
          self.placeholders['num_vertices']: bucket_sizes[bucket],
          self.placeholders['adjacency_matrix']: batch_data['adj_mat'],
          self.placeholders['node_mask']: batch_data['node_mask'],
          self.placeholders['graph_state_keep_prob']: dropout_keep_prob,
      }
      bucket_counters[bucket] += 1
      yield batch_feed_dict

  def evaluation(self):
    outputs = self.eval(ops=[self.predictions, self.targets,
                             self.placeholders['adjacency_matrix'],
                             self.placeholders['node_mask']])
    output_data = []
    for p, t, adj, mask in zip(*[np.split(o, len(o)) for o in outputs]):
      n_nodes = int(np.round(np.sum(mask)))
      output_data.append({
          'adjacency_matrix': np.squeeze(adj)[:n_nodes, :n_nodes].tolist(),
          'target': t.tolist(),
          'prediction': p[:, :n_nodes].tolist(),
      })
    with open(self.arguments.evaluation_file, 'w') as fp:
      json.dump(output_data, fp)


def main(args):
  model = DenseGraphNet(args)
  if args.evaluation_file:
    model.evaluation()
  else:
    model.train()


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Trains a dense graph net')
  parser.add_argument('--evaluation_file', action='store', default='', help='Path where the evaluation json is stored')
  DenseGraphNet.add_arguments(parser)
  args = parser.parse_args()
  main(args)
