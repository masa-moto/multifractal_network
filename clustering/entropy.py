import networkx as nx 
import numpy as np 
from numba import njit


@njit
def _GE_calc(boundary_degree, boundary_connections):
    entropy = 0.0
    for i in range(len(boundary_degree)):
        if boundary_degree[i] > 0 and boundary_connections[i] > 0:  # Add safety check
            p1 = boundary_connections[i] / boundary_degree[i]
            p0 = 1.0 - p1
            
            # Inline entropy calculation
            if p1 > 0:
                entropy -= p1 * np.log(p1)
            if p0 > 0:
                entropy -= p0 * np.log(p0)
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
    