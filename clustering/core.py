"""
this file is for core functions about entropy based clustering.
"""
import numpy as np 
import networkx as nx 
from numba import njit
from typing import List, FrozenSet, Set 
from concurrent.futures import ThreadPoolExecutor
from collections import deque


from .entropy import _graph_entropy_calc


def _update_boundary(graph, cluster):
    boundary_nodes_set = {
        neighbour for node in cluster
        for neighbour in graph.neighbors(node)
        if neighbour not in cluster
    }
    deg_dict = dict(graph.degree)
    return sorted(
        [(b_node, deg_dict[b_node]) for b_node in boundary_nodes_set],
        key = lambda x:x[1],
        reverse=True)


def _update_cluster_internal(graph:nx.Graph, cluster:nx.Graph, seed:int|str) -> Set:
    """
    update cluster and calculate GE recurrently so that GE to be minimized. only internal nodes of cluster are considered.

    Args:
        graph (nx.Graph): graph object including "cluster" as subgraph
        cluster (nx.Graph): cluster on the "graph". cluster must be connected, include at less than one node.

    Returns:
        FrozenSet: the cluster that minimize GE. only one cluster is considered.
    """
    # candidates are consisted of internal nodes of cluster excluding seed node.    
    deg_dict = dict(graph.degree)
    candidates = deque(
            sorted(
                [(idx, deg_dict[idx]) for idx in cluster if idx != seed],
                key=lambda x: x[1],
                reverse=True
                )
            )
    while candidates:
        #calculate GE of current cluster.
        previous_GE = _graph_entropy_calc(graph, cluster)
        
        #pick one candidate node of cluster neighbour. 
        idx, deg = candidates.popleft()
        
        # create temporal cluster
        poped_cluster = cluster - {idx}
        if not poped_cluster:
            continue
        #recalculate GE of the updated cluster.
        posterious_GE = _graph_entropy_calc(graph, poped_cluster)
        # delete the candidate from cluster.
        if posterious_GE < previous_GE:
            cluster = poped_cluster
        # when deque becomes empty, while-loop ends automatically     
    cluster |= {seed}
    return cluster

def _update_cluster_boundary(graph:nx.Graph, cluster:nx.Graph, seed) -> FrozenSet:
    """
    update cluster and calculate GE recurrently so that GE to be minimized. only boundary nodes of cluster are considered.

    Args:
        graph (nx.Graph): graph object including "cluster" as subgraph
        cluster (nx.Graph or Set): cluster on the "graph". cluster must be connected, include at less than one node.
        cutoff (float): cutoff threshold of GE. in case GE become less than cutoff, while loop ends.

    Returns:
        FrozenSet: the cluster that minimize GE. only one cluster is considered.
    """
    cluster = set(cluster) if not isinstance(cluster, set) else cluster
    candidates = _update_boundary(graph, cluster)
    previous_GE = _graph_entropy_calc(graph, cluster)
    GE_delta = 1
    while candidates and GE_delta < 0:
        
        GE_delta = float("inf")
        best_node = None
        
        # calculate GE of current cluster.
        
        # pick one candidate node of cluster neighbour.  
        # As a greedy algorithm, the candidate to be merged into the cluster is selected from all outer boundary nodes
        for node, deg in candidates:
            new_GE = _graph_entropy_calc(graph, cluster | {node})
            d = new_GE - previous_GE
            if d < GE_delta:
                best_node = node
                GE_delta = d

        if GE_delta < 0:
            cluster.add(best_node)
            previous_GE += GE_delta
            candidates = _update_boundary(graph, cluster)
        else:
            # print(f"{seed:4}, {previous_GE:.3f}, {len(cluster)}")
            break
        # when deque becomes empty, while-loop ends automatically       
    return cluster

def update_cluster(graph:nx.Graph, cluster:nx.Graph, seed:int, update_scope = "boundary"):
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
        assert cluster, f"empty cluster seed = {seed}"
        return _update_cluster_boundary(graph, cluster,seed)
    elif update_scope == "internal":
        assert cluster, f"empty cluster seed = {seed}"
        return _update_cluster_internal(graph, cluster,seed)
    else:
        raise ValueError(f"undefined scope: {update_scope}")
    
    
def seed_sorter(order, clusters):
    """sorting seed nodes of given cluster.

    Args:
        order (str, list, tuple): order of seed. default is descending order. "descending", "ascending", and iterable of seeds are acceptable.
        clusters (dict): dict object containing seed for its key and cluster for corresponding value.

    Raises:
        ValueError: if invalid order is input, raise ValueError

    Returns:
        seeds (list): list object for seeds. 
    """
    if order is None or order == "descending":
        seeds = sorted(clusters.keys(), key = lambda s:len(clusters[s]), reverse=True)
    elif order == "ascending":
        seeds = sorted(clusters.keys(), key = lambda s:len(clusters[s]), reverse=False)
    elif isinstance(order, (list, tuple)):
        seeds = [s for s in order if s in clusters]
    else:
        raise ValueError(f"unknown order : {order}")
    return seeds