from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import networkx as nx
import random
import tqdm


# http://oeis.org/A001349
_CONNECTED_GRAPHS = [1, 1, 1, 2, 6, 21, 112, 853, 11117, 261080, 11716571, 1006700565, 164059830476]


def random_graph(min_node, max_node):
  n_nodes = random.randint(min_node, max_node)
  gr = nx.generators.scale_free_graph(n_nodes)
  # This graph is directed, so we create an undirected version of it.
  # It also contains self-loops and duplicated edges.
  edges = set()
  for edge in gr.edges():
    if edge[0] == edge[1]:
      continue
    if (edge[1], edge[0]) in edges:
      continue
    edges.add(edge)
  graph = nx.Graph()
  graph.add_edges_from(edges)
  return graph


def jsonify(graph, target):
  nodes = [[0]] * len(graph.nodes())
  edges = []
  for edge in graph.edges():
    # start (0-indexed), type (1-indexed), end (0-indexed).
    edges.append([edge[0], 1, edge[1]])
  return {
      'targets': [[target]],
      'graph': edges,
      'node_features': nodes
  }


def main(args):
  data = []
  for _ in tqdm.tqdm(xrange(args.n_graphs)):
    graph = random_graph(args.min_nodes, args.max_nodes)
    target = nx.algebraic_connectivity(graph, method='lobpcg')
    data.append(jsonify(graph, target))
  with open(args.output_file, 'w') as fp:
    json.dump(data, fp)


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Generates random graph')
  parser.add_argument('--output_file', action='store', required=True, help='Path where the json file is stored')
  parser.add_argument('--min_nodes', type=int, action='store', default=8, help='Minimum number of nodes')
  parser.add_argument('--max_nodes', type=int, action='store', default=10, help='Maximum number of nodes')
  parser.add_argument('--n_graphs', type=int, action='store', default=10000, help='Number of graphs')
  args = parser.parse_args()
  main(args)
