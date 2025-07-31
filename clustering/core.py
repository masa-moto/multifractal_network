import numpy as np 
from numba import njit
import networkx as nx 
from concurrent.futures import ThreadPoolExecutor

_share_graph = None

def graph_initializer(graph):
    global _share_graph
    _share_graph = graph


def _update_cluster_inside(graph, cluster):
    pass

def update_cluster(graph, cluster, update_scope = "boundary", cutoff = 1e-10):
    """
    Update specific cluster of the graph according to the GE.
    Candidates are examined in descending order of their degrees.

    Args:
        graph (nx.Graph): graph to be examined.
        cluster (set): set of nodes included in cluster which must be the subgraph of the graph input.
        candidates(str): criteria of selecting candidates. "boundary" and "internal" is acceptable.
        cutoff (float): threshold of GE-delta for updating cluster
    Return:
        cluster (FrozenSet): frozenset of nodes consisted of updated cluster.
    """
    if update_scope == "boundary":
        _update_cluster_boundary(graph, cluster, cutoff)
    elif update_scope == "internal":
        pass 
    else:
        raise ValueError(f"undefined scope: {update_scope}")
    GE_delta = 100
    # while GE_delta > cutoff:
        #calculate GE of current cluster.
        #choosing candidate of cluster neighbour.
        #recalculate GE of the updated cluster.
        #examine if GE descend according to the cluster update
            #branch A; candidate joined: update candidates in descending order of their degree
            #reload candidates since neighbour of cluster changes
            
            #branch B; candidate do not join: look for the next candidate
            #while loop continues
        
    pass 