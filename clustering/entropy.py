import networkx as nx 
import numpy as np 
from numba import njit
from typing import Set, Iterable

try:
    from numba import njit 
except ImportError:
    def njit(func=None, **_k):
        return func if func else (lambda f: f)

@njit
def _GE_calc(boundary_degree, boundary_connections):
    entropy = 0.0
    for i in range(len(boundary_degree)):
        deg_i, conn_i = boundary_degree[i], boundary_connections[i]
        if deg_i > 0 and conn_i > 0:  # Add safety check
            p1 = conn_i / deg_i
            
        if p1<=0.0 or p1 >= 1.0:
            continue
            
        p0 = 1.0 - p1
            # Inline entropy calculation
        entropy -= p1 * np.log(p1) - p0 * np.log(p0)
    return entropy
    
    
def _graph_entropy_calc(graph: nx.Graph, cluster:Iterable) -> float:
    """
    calculate GE(graph entropy) of given cluster on the graph.
    not manipulate cluster growing process.
    this function only calculate GE on the graph with the given cluster.

    Args:
        graph (nx.Graph): graph object.
        cluster (nx.Graph): subgraph of the graph.
    Return:
        entropy (float): GE for the given pair of graph and cluster
    """
    if not cluster:
        return 0.0
    
    # Convert cluster to set if it's not already
    cluster_nodes = set(cluster) if not isinstance(cluster, set) else cluster
    
    # Get boundary nodes more efficiently using set comprehension
    boundary_nodes_set = {
        neighbor 
        for node in cluster_nodes 
        for neighbor in graph.neighbors(node) 
        if neighbor not in cluster_nodes
    }
    
    if not boundary_nodes_set:  # Early return for empty boundary
        return 0.0
    
    # Calculate entropy for boundary nodes
    entropy = 0.0
    boundary_nodes = list(boundary_nodes_set)
    boundary_degree = np.array([graph.degree[node] for node in boundary_nodes], dtype = np.float64)
    boundary_connections = np.array(
        [sum(1 for neighbor in graph.neighbors(node) if neighbor in cluster_nodes)
        for node in boundary_nodes],
        dtype = np.float64)
    return _GE_calc(boundary_degree, boundary_connections)
    