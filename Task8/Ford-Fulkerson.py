import networkx as nx
import numpy as np
import random
from itertools import permutations
import timeit
from time import time
import matplotlib.pyplot as plt


def random_graph(n, m):
    graph = nx.Graph()
    N_range = range(n)
    graph.add_nodes_from(N_range)
    for pair in random.sample([*permutations(N_range, 2)], m):
        graph.add_edge(*pair, weight=random.randint(0, 15))
    return graph


n = 20
m = 150
graph1 = random_graph(n, m)
nx.draw_circular(graph1, node_color='red', node_size=200, with_labels=True)
weights = nx.get_edge_attributes(graph1, 'weight')
pos = nx.circular_layout(graph1)
# nx.draw_networkx_edge_labels(graph1, pos, edge_labels=weights)
matrix = nx.adjacency_matrix(graph1).todense()
matrix2 = nx.adjacency_matrix(graph1)
graphmatrix = matrix.tolist()
print(matrix2.todense())
plt.savefig('graph.png', dpi=300)


def fs(s, t, parent):
    visited = [False] * n
    queue = [s]
    visited[s] = True
    while queue:
        u = queue.pop(0)
        for ind, val in enumerate(graphmatrix[u]):
            if visited[ind] is False and val > 0:
                queue.append(ind)
                visited[ind] = True
                parent[ind] = u
    return True if visited[t] else False


def fl(graphmatrix, source, sink):
    parent = [-1] * n
    max_flow = 0
    while fs(source, sink, parent):
        path_flow = float("Inf")
        s = sink
        while s != source:
            path_flow = min(path_flow, graphmatrix[parent[s]][s])
            s = parent[s]
            max_flow += path_flow
            v = sink
        while v != source:
            u = parent[v]
            graphmatrix[u][v] -= path_flow
            graphmatrix[v][u] += path_flow
            v = parent[v]
    return max_flow


source = 0
sink = 10
t1 = time()
print("The maximum flow from node", source, "to node", sink, "is", fl(graphmatrix, source, sink))
t2 = time()
print('Execution time', t2 - t1)
