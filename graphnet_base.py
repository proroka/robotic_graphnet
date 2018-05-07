from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import json
import numpy as np
import os
import pickle
import random
import sys
import tensorflow as tf
import time

import utils


class GraphNetBase(object):
  __metaclass__ = abc.ABCMeta

  @classmethod
  def default_params(cls):
    return {
        'num_epochs': 5,   # Number of epochs.
        'patience': 25,    # Number of epochs without improvement.
        'task_ids': [0],   # To predict multiple targets.
        'task_sample_ratios': {},  # Downweight more frequent tasks.
        'random_seed': 0,  # Random seed.

        'learning_rate': 0.001,
        'clamp_gradient_norm': 1.0,
        'out_layer_dropout_keep_prob': 1.0,
        'hidden_size': 100,
        'num_timesteps': 4,    # Number of timesteps to unroll message passing.
        'use_graph': True,     # Whether to even propagate messsages.
        'tie_fwd_bkwd': True,  # Combine forward and backward edges.

        'train_file': 'train8_10.json',
        'valid_file': 'eval8_10.json'
    }

  @classmethod
  def add_arguments(cls, parser):
    parser.add_argument('--data_directory', action='store', default='data', help='Path where the json data files are stored')
    parser.add_argument('--train_file', action='store', default='train8_10.json', help='Relative path from data directory')
    parser.add_argument('--valid_file', action='store', default='eval8_10.json', help='Relative path from data directory')
    parser.add_argument('--log_directory', action='store', default='logs', help='Path where the logs are stored')
    parser.add_argument('--restore', action='store', default=None, help='Path where the logs are stored')
    parser.add_argument('--num_epochs', type=int, action='store', default=5, help='Number of epochs')
    parser.add_argument('--num_timesteps', type=int, action='store', default=4, help='Number of timesteps')
    parser.add_argument('--restrict_data', type=int, action='store', default=0, help='Truncate data')
    parser.add_argument('--freeze_graph_model', action='store_true', help='Do not update the graph model')

  def __init__(self, args):
    # Command-line arguments (using argparse).
    self._args = args

    # Location of the data files.
    self._data_dir = args.data_directory
    self._run_id = "_".join([time.strftime("%Y-%m-%d-%H-%M-%S"), str(os.getpid())])
    self._log_file = os.path.join(args.log_directory, '%s_log.json' % self._run_id)
    self._best_model_file = os.path.join(args.log_directory, '%s_model_best.pickle' % self._run_id)

    # Other parameters.
    self._params = self.default_params()
    self._params['train_file'] = args.train_file
    self._params['valid_file'] = args.valid_file
    self._params['num_epochs'] = args.num_epochs
    self._params['num_timesteps'] = args.num_timesteps
    self.set_arguments(self._args, self._params)
    with open(os.path.join(args.log_directory, '%s_params.json' % self._run_id), 'w') as f:
      json.dump(self._params, f)
    print('Run %s starting with following parameters:\n%s' % (self._run_id, json.dumps(self._params)))
    random.seed(self._params['random_seed'])
    np.random.seed(self._params['random_seed'])

    # Load data.
    train_data, train_n_vertices, train_n_edge_types, train_n_annotation = self.load_data(
        self._params['train_file'], restrict_data=self._args.restrict_data)
    valid_data, valid_n_vertices, valid_n_edge_types, valid_n_annotation = self.load_data(
        self._params['valid_file'], restrict_data=self._args.restrict_data)
    self._max_num_vertices = max(train_n_vertices, valid_n_vertices)
    self._num_edge_types = max(train_n_edge_types, valid_n_edge_types)
    self._annotation_size = max(train_n_annotation, valid_n_annotation)
    self._train_data = self.process_raw_graphs(train_data, is_training_data=True)
    self._valid_data = self.process_raw_graphs(valid_data, is_training_data=False)

    # Build the actual model
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    self._graph = tf.Graph()
    self._session = tf.Session(graph=self._graph, config=config)
    with self._graph.as_default():
      tf.set_random_seed(self._params['random_seed'])
      self._placeholders = {}
      self._weights = {}
      self._ops = {}
      self.make_model()
      self.make_train_step()
      # Restore or initialize variables.
      restore_file = args.restore
      if restore_file is not None:
        self.restore_model(restore_file)
      else:
        self.initialize_model()

  def load_data(self, filename, restrict_data=0):
    full_path = os.path.join(self._data_dir, filename)
    print('Loading data from %s' % full_path)
    with open(full_path, 'r') as fp:
      data = json.load(fp)
    if restrict_data:
      data = data[:restrict_data]

    # General data.
    max_num_vertices = 0
    num_fwd_edge_types = 0
    for g in data:
      max_num_vertices = max(max_num_vertices, 1 + max(v for e in g['graph'] for v in [e[0], e[2]]))
      num_fwd_edge_types = max(num_fwd_edge_types, max(e[1] for e in g['graph']))
    num_edge_types = num_fwd_edge_types * (1 if self._params['tie_fwd_bkwd'] else 2)
    annotation_size = len(data[0]['node_features'][0])
    return data, max_num_vertices, num_edge_types, annotation_size

  @staticmethod
  def graph_string_to_array(graph_string):
    return [[int(v) for v in s.split(' ')] for s in graph_string.split('\n')]

  @abc.abstractmethod
  def set_arguments(self, args, params):
      """Adds parameters."""

  @property
  def parameters(self):
    return self._params

  @property
  def arguments(self):
    return self._args

  @property
  def session(self):
    return self._session

  @property
  def num_edge_types(self):
    return self._num_edge_types

  @property
  def max_num_vertices(self):
    return self._max_num_vertices

  @property
  def annotation_size(self):
    return self._annotation_size

  @property
  def placeholders(self):
    return self._placeholders

  @property
  def weights(self):
    return self._weights

  @property
  def predictions(self):
    return self._predictions

  @property
  def targets(self):
    return self._targets

  @abc.abstractmethod
  def process_raw_graphs(self, raw_data, is_training_data):
      """Process JSON graph data."""

  def make_model(self):
    # Create placeholders.
    self._placeholders['target_values'] = tf.placeholder(tf.float32, [len(self._params['task_ids']), None], name='target_values')
    self._placeholders['target_mask'] = tf.placeholder(tf.float32, [len(self._params['task_ids']), None], name='target_mask')
    self._placeholders['num_graphs'] = tf.placeholder(tf.int64, [], name='num_graphs')
    self._placeholders['out_layer_dropout_keep_prob'] = tf.placeholder(tf.float32, [], name='out_layer_dropout_keep_prob')

    with tf.variable_scope('graph_model'):
      self.prepare_specific_graph_model()
      # This does the actual graph work.
      if self._params['use_graph']:
        self._ops['final_node_representations'] = self.compute_final_node_representations()
      else:
        self._ops['final_node_representations'] = tf.zeros_like(self._placeholders['initial_node_representation'])

    # Compute the squared loss and absolute difference for each prediction task.
    self._ops['losses'] = []
    for (internal_id, task_id) in enumerate(self._params['task_ids']):
      with tf.variable_scope('out_layer_task_%d' % task_id):
        with tf.variable_scope('regression_gate'):
          self._weights['regression_gate_task_%d' % task_id] = utils.MLP(
              2 * self._params['hidden_size'], 1, [], self._placeholders['out_layer_dropout_keep_prob'])
        with tf.variable_scope('regression'):
          self._weights['regression_transform_task_%d' % task_id] = utils.MLP(
              self._params['hidden_size'], 1, [], self._placeholders['out_layer_dropout_keep_prob'])
        computed_values = self.gated_regression(self._ops['final_node_representations'],
                                                self._weights['regression_gate_task_%d' % task_id],
                                                self._weights['regression_transform_task_%d' % task_id])
        diff = computed_values - self._placeholders['target_values'][internal_id, :]
        task_target_mask = self._placeholders['target_mask'][internal_id, :]
        task_target_num = tf.reduce_sum(task_target_mask) + utils.SMALL_NUMBER
        diff = diff * task_target_mask  # Mask out unused values
        self._ops['accuracy_task_%d' % task_id] = tf.reduce_sum(tf.abs(diff)) / task_target_num
        self._predictions = computed_values
        self._targets = self._placeholders['target_values'][internal_id, :]
        task_loss = tf.reduce_sum(0.5 * tf.square(diff)) / task_target_num
        # Normalise loss to account for fewer task-specific examples in batch:
        task_loss = task_loss * (1.0 / (self._params['task_sample_ratios'].get(task_id) or 1.0))
        self._ops['losses'].append(task_loss)
    self._ops['loss'] = tf.reduce_sum(self._ops['losses'])

  def make_train_step(self):
    trainable_vars = self._session.graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    if self._args.freeze_graph_model:
      graph_vars = set(self._session.graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='graph_model'))
      filtered_vars = []
      for var in trainable_vars:
        if var not in graph_vars:
          filtered_vars.append(var)
        else:
          print('Freezing weights of variable %s.' % var.name)
      trainable_vars = filtered_vars
    # Optimize (and clip gradients).
    optimizer = tf.train.AdamOptimizer(self._params['learning_rate'])
    grads_and_vars = optimizer.compute_gradients(self._ops['loss'], var_list=trainable_vars)
    clipped_grads = []
    for grad, var in grads_and_vars:
      if grad is not None:
        clipped_grads.append((tf.clip_by_norm(grad, self._params['clamp_gradient_norm']), var))
      else:
        clipped_grads.append((grad, var))
    self._ops['train_step'] = optimizer.apply_gradients(clipped_grads)
    # Initialize newly-introduced variables:
    # self._session.run(tf.local_variables_initializer())

  @abc.abstractmethod
  def gated_regression(self, last_h, regression_gate, regression_transform):
    """Gated regression."""

  @abc.abstractmethod
  def prepare_specific_graph_model(self):
    """Graph model."""

  @abc.abstractmethod
  def compute_final_node_representations(self):
    """Compute node representations."""

  @abc.abstractmethod
  def make_minibatch_iterator(self, data, is_training):
    """Create mini-batch iterator."""

  def run_epoch(self, epoch_name, data, is_training, ops=None):
    loss = 0
    accuracies = []
    op_results = []
    accuracy_ops = [self._ops['accuracy_task_%d' % task_id] for task_id in self._params['task_ids']]
    start_time = time.time()
    processed_graphs = 0
    batch_iterator = utils.ThreadedIterator(self.make_minibatch_iterator(data, is_training), max_queue_size=5)
    for step, batch_data in enumerate(batch_iterator):
      num_graphs = batch_data[self._placeholders['num_graphs']]
      processed_graphs += num_graphs
      if is_training:
        batch_data[self._placeholders['out_layer_dropout_keep_prob']] = self._params['out_layer_dropout_keep_prob']
        fetch_list = [self._ops['loss'], accuracy_ops, self._ops['train_step']]
      else:
        batch_data[self._placeholders['out_layer_dropout_keep_prob']] = 1.0
        fetch_list = [self._ops['loss'], accuracy_ops]
        if ops is not None:
          fetch_list.extend(ops)
      result = self._session.run(fetch_list, feed_dict=batch_data)
      batch_loss, batch_accuracies = result[:2]
      if not is_training and ops is not None:
        op_results.append(result[2:])
      loss += batch_loss * num_graphs
      accuracies.append(np.array(batch_accuracies) * num_graphs)
      print('Running %s, batch %d (has %d graphs). Loss so far: %.4f' % (
          epoch_name, step, num_graphs, loss / (processed_graphs + utils.SMALL_NUMBER)), end='\r')
      sys.stdout.flush()
    if accuracies:
      accuracies = np.sum(accuracies, axis=0) / (processed_graphs + utils.SMALL_NUMBER)
    loss /= (processed_graphs + utils.SMALL_NUMBER)
    instance_per_sec = processed_graphs / (time.time() - start_time)
    if ops is not None:
      batched_values = zip(*op_results)
      op_results = []
      for values in batched_values:
        op_results.append(np.concatenate(values, axis=0))  # Batch concatenation.
      return loss, accuracies, instance_per_sec, op_results
    return loss, accuracies, instance_per_sec

  def eval(self, ops=None):
    valid_loss, valid_accs, valid_speed, outputs = self.run_epoch('Evaluation', self._valid_data, False, ops=ops)
    accs_str = ' '.join(['%d:%.5f' % (id, acc) for (id, acc) in zip(self._params['task_ids'], valid_accs)])
    print('\r\x1b[K Valid: loss: %.5f | acc: %s | instances/sec: %.2f' % (valid_loss, accs_str, valid_speed))
    return outputs

  def train(self):
    log_to_save = []
    total_time_start = time.time()
    with self._graph.as_default():
      if self._args.restore:
        _, valid_accs, _ = self.run_epoch('Resumed (validation)', self._valid_data, False)
        best_val_acc = np.sum(valid_accs)
        best_val_acc_epoch = 0
        print('\r\x1b[KResumed operation, initial cum. val. acc: %.5f' % best_val_acc)
      else:
        (best_val_acc, best_val_acc_epoch) = (float('+inf'), 0)
      for epoch in range(1, self._params['num_epochs'] + 1):
        print('== Epoch %d' % epoch)
        train_loss, train_accs, train_speed = self.run_epoch(
            'epoch %d (training)' % epoch, self._train_data, True)
        accs_str = ' '.join(['%d:%.5f' % (id, acc) for (id, acc) in zip(self._params['task_ids'], train_accs)])
        print('\r\x1b[K Train: loss: %.5f | acc: %s | instances/sec: %.2f' % (
            train_loss, accs_str, train_speed))
        valid_loss, valid_accs, valid_speed = self.run_epoch(
            'epoch %d (validation)' % epoch, self._valid_data, False)
        accs_str = ' '.join(['%d:%.5f' % (id, acc) for (id, acc) in zip(self._params['task_ids'], valid_accs)])
        print('\r\x1b[K Valid: loss: %.5f | acc: %s | instances/sec: %.2f' % (
            valid_loss, accs_str, valid_speed))
        epoch_time = time.time() - total_time_start
        log_entry = {
            'epoch': epoch,
            'time': epoch_time,
            'train_results': (train_loss, train_accs.tolist(), train_speed),
            'valid_results': (valid_loss, valid_accs.tolist(), valid_speed),
        }
        log_to_save.append(log_entry)
        with open(self._log_file, 'w') as f:
          json.dump(log_to_save, f, indent=4)

        val_acc = np.sum(valid_accs)
        if val_acc < best_val_acc:
          self.save_model(self._best_model_file)
          print('  (Best epoch so far, cum. val. acc decreased to %.5f from %.5f. Saving to \'%s\')' % (val_acc, best_val_acc, self._best_model_file))
          best_val_acc = val_acc
          best_val_acc_epoch = epoch
        elif epoch - best_val_acc_epoch >= self._params['patience']:
          print('Stopping training after %d epochs without improvement on validation accuracy.' % self._params['patience'])
          break

  def save_model(self, path):
    weights_to_save = {}
    for variable in self._session.graph.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
      assert variable.name not in weights_to_save
      weights_to_save[variable.name] = self._session.run(variable)
    data_to_save = {
        'params': self._params,
        'weights': weights_to_save
    }
    with open(path, 'wb') as out_file:
      pickle.dump(data_to_save, out_file, pickle.HIGHEST_PROTOCOL)

  def initialize_model(self):
    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())
    self._session.run(init_op)

  def restore_model(self, path):
    print('Restoring weights from file %s.' % path)
    with open(path, 'rb') as in_file:
      data_to_load = pickle.load(in_file)
    # Assert that we got the same model configuration
    assert len(self._params) == len(data_to_load['params'])
    for (par, par_value) in self._params.items():
      # Fine to have different task_ids:
      if par not in ['task_ids', 'num_epochs', 'valid_file', 'train_file', 'batch_size']:
        assert par_value == data_to_load['params'][par]
    variables_to_initialize = []
    with tf.name_scope('restore'):
      restore_ops = []
      used_vars = set()
      for variable in self._session.graph.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
        used_vars.add(variable.name)
        if variable.name in data_to_load['weights']:
          restore_ops.append(variable.assign(data_to_load['weights'][variable.name]))
        else:
          print('Freshly initializing %s since no saved value was found.' % variable.name)
          variables_to_initialize.append(variable)
      for var_name in data_to_load['weights']:
        if var_name not in used_vars:
          print('Saved weights for %s not used by model.' % var_name)
      restore_ops.append(tf.variables_initializer(variables_to_initialize))
      self._session.run(restore_ops)
