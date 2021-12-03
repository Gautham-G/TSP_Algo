import networkx as nx
import numpy as np
import pandas as pd
from scipy.spatial import distance_matrix
from queue import PriorityQueue
import time
from Approx import *

def BnB(G, max_time):
  start = time.time()
  trace=[]
  # initial approximate solution
  approx_dist, approx_tour, approx_trace, approx_runtime = Approx(G,max_time)
  q = PriorityQueue()
  start_node=0
  q.put((np.infty,[start_node]))
  best=(approx_dist,approx_tour)
  trace.append([approx_runtime, approx_dist])
  timeout_minutes = 20
  timeout = time.time() + max_time
  while not q.empty():
    if time.time() > timeout:
      break
    item = q.get()
    for node in G.nodes-item[1]:
      subtour=item[1].copy()
      subtour.append(node)
      if len(G.nodes-subtour)==0:
        # complete solution
        tour = subtour+[subtour[0]]
        cost = path_weight(G, tour, 'weight')
        if cost < best[0]:
          # new best solution, else dead end and do nothing
          best=(cost, tour)
          trace.append([round(time.time()-start, 2), best[0]])
      else:
        # incomplete solution
        subproblem=G.subgraph(G.nodes-subtour)
        mst_cost=nx.minimum_spanning_tree(subproblem).size(weight='weight')
        subtour_cost=path_weight(G,subtour,'weight')
        min_weight_edge_start = np.infty
        min_weight_edge_end = np.infty
        for node in subproblem:
          if G.get_edge_data(subtour[0],node)['weight'] < min_weight_edge_start:
            min_weight_edge_start = G.get_edge_data(subtour[0],node)['weight']
          if G.get_edge_data(subtour[-1],node)['weight'] < min_weight_edge_end:
            min_weight_edge_end = G.get_edge_data(subtour[-1],node)['weight']
        # lower bound cost of connecting subtour and subproblem
        connection_cost = min_weight_edge_start + min_weight_edge_end
        # lower bound
        cost = mst_cost + subtour_cost + connection_cost
        if cost < best[0]:
          # new subproblem
          q.put((cost,subtour))
  dist, tour = best
  runtime = trace[-1][0] if q.empty() else max_time
  return dist, tour, trace, runtime
