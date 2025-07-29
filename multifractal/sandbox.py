import networkx as nx 
import numpy as np 
import random as rd 
import scipy as cp 
from numba import njit 
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import cpu_count
from typing import List, Tuple, Dict, Sequence


def compute_sandbox_measure(graph_csr, N, source, scale)->Sequence[int]:
    order, pred = cp.sparse.csgraph.breadth_first_order(
        graph_csr, source, directed = False, return_predecessors = True
    )
    """
    Compute the number of nodes within distance r (0 ≤ r < diam)
    from the given source in the graph represented by graph_csr.
    Returns a tuple of length diam, where each element is the count of nodes at distance ≤ r.
    """
    dist = np.full(N, -1, dtype=int)
    dist[source] = 0
    for idx in order[1:]:
        if pred[idx] >= 0:
            dist[idx] = dist[pred[idx]] + 1
    return np.array([np.count_nonzero(dist<=r) for r in scale], dtype=float)#距離r以内のノード数
    # return len(set(nx.single_source_shortest_path_length(graph, source, cutoff=radius).keys()))

def compute_sandbox(graph_csr, scale, diam, source):
    # print(f"This is in sandbox.py/compute_sandbox(), args are follows,\n! ommit graph, scale:{scale}, diam:{diam}, source:{source}")
    N = graph_csr.shape[0]
    mu_array = compute_sandbox_measure(
                    graph_csr=graph_csr,
                    N=N,
                    source=source,
                    scale = scale
                )
    # print(f"this is from sandbox.py. : {mu_array}")
    # print(f"{scale}/{diam}")
    return mu_array/N