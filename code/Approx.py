import networkx as nx
import numpy as np
import pandas as pd
from scipy.spatial import distance_matrix
import time

def path_weight(G, path, weight):
#     multigraph = G.is_multigraph()
    cost = 0

    if not nx.is_path(G, path):
        raise nx.NetworkXNoPath("path does not exist")
    for node, nbr in nx.utils.pairwise(path):
#         if multigraph:
#             cost += min([v[weight] for v in G[node][nbr].values()])

        cost += G[node][nbr][weight]
    return cost
def Approx(G, max_time):
  start = time.time()
  # MST via Kruskal's algorithm
  mst=nx.minimum_spanning_tree(G)
  # eulerian cycle over directed symmetrical edges of MST
  eulerian_cycle = list(nx.eulerian_circuit(mst.to_directed(),source=0))
  visited_nodes=[]
  tour_nodes=[]
  for edge in eulerian_cycle:
    # shortcut over vertices in the eulerian cycle seen once already
    if(edge[0] not in visited_nodes):
      tour_nodes.append(edge[0])
    visited_nodes.append(edge[0])
  # complete the shortcutted cycle
  tour_nodes.append(tour_nodes[0])
  finish = time.time()
  dist = path_weight(G,tour_nodes, 'weight')
  tour = tour_nodes
  trace = [[round(finish-start, 2), dist]]
  runtime = min(trace[-1][0], max_time)
  return dist, tour, trace, runtime
