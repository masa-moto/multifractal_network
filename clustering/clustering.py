import networkx as nx 
import numpy as np 
import multiprocessing as mp 
from typing import FrozenSet, List, Set

import entropy
import core 



_share_graph = nx.Graph()
_share_threshold = 1e-10

def graph_initializer(graph:nx.Graph, threshold:float):
    global _share_graph
    global _share_threshold
    _share_graph = graph
    _share_threshold = threshold


def _update_cluster_wrapper(args):
    
    cluster, seed = args
    # initial cluster processing...
    cluster = core.update_cluster(_share_graph, set(cluster), seed, "internal", cutoff = _share_threshold)
    
    # final cluster processing...
    return seed, core.update_cluster(_share_graph, cluster, seed, "boundary", cutoff = _share_threshold)

def entropy_based_clustering(
    graph:nx.Graph,
    cluster_cutoff_size:int = 2,
    GE_threshold:float = 1e-3,
    ) -> List[Set]:
    """
    find GE(graph entropy) based clustering. this function uses algorithm by C.K.Edward and C.Young-Rae (doi:10.1109/ICDM.2011.64)

    Args:
        grpah (nx.Graph): graph to be examined.

    Returns:
        List[FrozenSet]: index of return represents seed nodes and FrozenSet represents GE based cluster of corresponding seed node.
    """
    
    

    # create initial cluster consisted of seed and its neighbours
    nodes = graph.nodes
    graph_csr = nx.to_scipy_sparse_array(graph, nodelist=nodes, format = "csr")
    init_clusters = [set(map(int, graph_csr[n:n+1,:].nonzero()[1])) for n in range(graph_csr.shape[0])]
    
    # see if there is any candidate to delete from cluster to minimize GE
    init_args = (graph, GE_threshold)
    args = zip(init_clusters, nodes)
    with mp.Pool(processes= mp.cpu_count(), initializer=graph_initializer, initargs=init_args) as p:
        clusters = p.map(_update_cluster_wrapper, args)
    
    # filtering cluster by their size
    clusters = [(seed, cluster) for seed, cluster in clusters if len(cluster) > cluster_cutoff_size]
    return clusters

def modularity_based_clustering(graph:nx.Graph):
    return nx.greedy_modularity_communities(graph)