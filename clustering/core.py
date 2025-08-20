"""
this file is for core functions about entropy based clustering.
"""
import numpy as np 
import networkx as nx 
from typing import List, FrozenSet, Set, Dict, Iterable
from collections import deque


from .entropy import _graph_entropy_calc


def _update_boundary(graph:nx.Graph, cluster:Set, deg_dict:Dict):
    boundary_nodes_set = {
        neighbour for node in cluster
        for neighbour in graph.neighbors(node)
        if neighbour not in cluster
    }
    return sorted(
        [(b_node, deg_dict[b_node]) for b_node in boundary_nodes_set],
        key = lambda x:x[1],
        reverse=True)


def _update_cluster_internal(graph:nx.Graph, cluster:nx.Graph | Set, seed:int|str, deg_dict: Dict|None) -> Set:
    """
    update cluster and calculate GE recurrently so that GE to be minimized. only internal nodes of cluster are considered.

    Args:
        graph (nx.Graph): graph object including "cluster" as subgraph
        cluster (nx.Graph): cluster on the "graph". cluster must be connected, include at less than one node.
        cutoff (float): cutoff threshold of GE. in case GE become less than cutoff, while loop ends.

    Returns:
        FrozenSet: the cluster that minimize GE. only one cluster is considered.
    """
    # candidates are consisted of internal nodes of cluster excluding seed node.    
    if not cluster:
        return cluster
    if deg_dict is None:
        deg_dict = dict(graph.degree)
    
    candidates = deque(
            sorted(
                [(idx, deg_dict[idx]) for idx in cluster],
                key=lambda x: x[1],
                reverse=True
                )
            )
    print(f"{__name__}: 1: {cluster}, {candidates}")
    
    previous_GE = _graph_entropy_calc(graph, cluster)
    while candidates:
        #calculate GE of current cluster.
        # previous_GE = _graph_entropy_calc(graph, cluster)
        
        #pick one candidate node of cluster neighbour. 
        idx, _deg = candidates.popleft()
        if idx == seed:
            continue
        
        # create temporal cluster
        poped_cluster = cluster - {idx}
        if not poped_cluster:
            continue
        #recalculate GE of the updated cluster.
        posterious_GE = _graph_entropy_calc(graph, poped_cluster)
        # delete the candidate from cluster.
        print(f"{__name__}: 2: {cluster}, {idx}, {previous_GE:.3f}, {posterious_GE:.3f}")
        if posterious_GE < previous_GE:
            cluster = poped_cluster
            previous_GE = posterious_GE
        # when deque becomes empty, while-loop ends automatically     
    return cluster

def _update_cluster_boundary(
    graph:nx.Graph,
    cluster:nx.Graph,
    seed,
    cutoff: float = 1e-5,
    deg_dict:None|Dict = None) -> Set:
    """
    update cluster and calculate GE recurrently so that GE to be minimized. only boundary nodes of cluster are considered.

    Args:
        graph (nx.Graph): graph object including "cluster" as subgraph
        cluster (nx.Graph or Set): cluster on the "graph". cluster must be connected, include at less than one node.
        cutoff (float): cutoff threshold of GE. in case GE become less than cutoff, while loop ends.

    Returns:
        FrozenSet: the cluster that minimize GE. only one cluster is considered.
    """
    if not cluster:
        return cluster
    if deg_dict is None:
        deg_dict = dict(graph.degree)

    cluster = set(cluster) if not isinstance(cluster, set) else cluster
    candidates = _update_boundary(graph, cluster, deg_dict)
    previous_GE = _graph_entropy_calc(graph, cluster)
    GE_delta = cutoff+1
    while candidates and abs(GE_delta) > cutoff:
        
        GE_delta = 0
        best_node = None
        
        # calculate GE of current cluster.
        
        # pick one candidate node of cluster neighbour.  
        # As a greedy algorithm, the candidate to be merged into the cluster is selected from all outer boundary nodes
        for node, _deg in candidates:
            new_GE = _graph_entropy_calc(graph, cluster | {node})
            delta = new_GE - previous_GE
            if delta < GE_delta:
                best_node = node
                GE_delta = delta
        if best_node is None or abs(GE_delta) <= cutoff:
            break
        
        cluster.add(best_node)
        previous_GE += GE_delta
        candidates = _update_boundary(graph, cluster, deg_dict)
        # when deque becomes empty, while-loop ends automatically       
    return cluster


def _update_cluster_boundary_0(
    graph:nx.Graph,
    cluster:nx.Graph,
    seed:int,
    cutoff: float = 1e-5,
    deg_dict:None|Dict = None) -> Set:
    """
    update cluster and calculate GE recurrently so that GE to be minimized. only boundary nodes of cluster are considered.
    Add node into cluster as soon as finding the node which decreases GE.
    Args:
        graph (nx.Graph): graph object including "cluster" as subgraph
        cluster (nx.Graph or Set): cluster on the "graph". cluster must be connected, include at less than one node.
        cutoff (float): cutoff threshold of GE. in case GE become less than cutoff, while loop ends.

    Returns:
        FrozenSet: the cluster that minimize GE. only one cluster is considered.
    """
    if not cluster:
        return cluster
    if deg_dict is None:
        deg_dict = dict(graph.degree)

    cluster = set(cluster) if not isinstance(cluster, set) else cluster
    candidates = deque(_update_boundary(graph, cluster, deg_dict))
    previous_GE = _graph_entropy_calc(graph, cluster)
    GE_delta = cutoff+1
    while candidates:    
        
        node, _deg = candidates.popleft()
        tentative_cluster = cluster | {node}
        new_GE = _graph_entropy_calc(graph, tentative_cluster)
        if new_GE < previous_GE:
            # if GE decreases, update cluster and candidate(boundary) of the cluster.
            cluster = tentative_cluster
            previous_GE = new_GE
            candidates = deque(_update_boundary(graph, cluster, deg_dict))
    return cluster
            



def update_cluster(
    graph:nx.Graph,
    cluster:nx.Graph,
    seed:int,
    update_scope = "boundary",
    cutoff = 1e-10,
    deg_dict:Dict | None = None,
    )-> Set:
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
    assert cluster, f"empty cluster; seed = {seed}"
    
    if deg_dict is None:
        deg_dict = dict(graph.degree)

    print(f"{__name__}: 3: {cluster} ")
    if update_scope == "boundary":
        assert cluster, f"empty cluster seed = {seed}"
        return _update_cluster_boundary_0(graph, cluster,seed, cutoff, deg_dict)
    elif update_scope == "internal":
        assert cluster, f"empty cluster seed = {seed}"
        return _update_cluster_internal(graph, cluster,seed, deg_dict)
    else:
        raise ValueError(f"undefined scope: {update_scope}")
    
    
def seed_sorter(order, clusters:Dict) -> List:
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
        return sorted(clusters.keys(), key = lambda s:len(clusters[s]), reverse=True)
    elif order == "ascending":
        return sorted(clusters.keys(), key = lambda s:len(clusters[s]), reverse=False)
    elif isinstance(order, (list, tuple)):
        return [s for s in order if s in clusters]
    raise ValueError(f"unknown order : {order}")