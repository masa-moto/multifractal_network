import networkx as nx 
import numpy as np 
from numba import njit


@njit
def _entropy_calc(p:np.float64)->np.float64:
    return -p*np.log(p) if p != 0 else 0

@njit
def _GE_calc(boundary_nodes, boundary_degree, boundary_connections):
    entropy = 0.0
    for i in range(len(boundary_nodes)):
        p1 = boundary_connections[i]/boundary_degree[i]
        p0 = 1-p1
        entropy+= _entropy_calc(p1) + _entropy_calc(p0)
    return entropy
    
    
def _graph_entropy_calc(graph, cluster):
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

    # Convert cluster to set if it's not already
    cluster_nodes = set(cluster) if not isinstance(cluster, set) else cluster
    # Get boundary nodes (neighbors of cluster nodes that are not in cluster)
    boundary_nodes = list()
    for node in cluster_nodes:
        for neighbor in graph.neighbors(node):
            if neighbor not in cluster_nodes:
                boundary_nodes.append(neighbor)
    
    # Calculate entropy for boundary nodes
    entropy = 0.0
    boundary_degree = list(graph.degree[node] for node in boundary_nodes)
    boundary_connections = list(sum(1 for neighbor in graph.neighbors(node) if neighbor in cluster_nodes) for node in boundary_nodes)
    assert (len(boundary_nodes) == len(boundary_degree)) and (len(boundary_degree) == len(boundary_connections)) and (len(boundary_connections) == len(boundary_nodes))
    return _GE_calc(boundary_nodes, boundary_degree, boundary_connections)
    