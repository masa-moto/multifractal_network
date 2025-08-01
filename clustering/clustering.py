import networkx as nx 
import numpy as np 
import multiprocessing as mp 
from typing import FrozenSet, List

import entropy
import core 



_share_graph = nx.Graph()

def graph_initializer(graph:nx.Graph, dum):
    global _share_graph
    _share_graph = graph


def _update_cluster_wrapper(args):
    
    cluster, seed, cutoff = args
    print(f"{seed}, ", end ="")
    # initial cluster processing...
    cluster = core.update_cluster(_share_graph, set(cluster), seed, "internal", cutoff = cutoff)
    
    # final cluster processing...
    return seed, core.update_cluster(_share_graph, cluster, seed, "boundary", cutoff = cutoff)

def entropy_based_clustering(graph:nx.Graph, threshold:float = 1e-3) -> List[FrozenSet]:
    """
    find GE(graph entropy) based clustering. this function uses algorithm by C.K.Edward and C.Young-Rae (doi:10.1109/ICDM.2011.64)

    Args:
        grpah (nx.Graph): graph to be examined.

    Returns:
        List[FrozenSet]: index of return represents seed nodes and FrozenSet represents GE based cluster of corresponding seed node.
    """
    
    nodes = graph.nodes()
    # create initial cluster consisted of seed and its neighbours
    print("init cluster making")
    init_clusters=[list(graph.neighbors(n)) for n in nodes]
    print("init cluster made.")
    
    # graph_csr = nx.to_scipy_sparse_array(graph, nodelist=nodes, format = "csr")
    
    # see if there is any candidate to delete from cluster to minimize GE
    init_args = (graph, "dum")
    args = zip(init_clusters, nodes)
    with mp.Pool(processes= mp.cpu_count(), initializer=graph_initializer, initargs=init_args) as p:
        clusters = p.map(_update_cluster_wrapper, args)
    # calculate GE and update cluster until GE becomes minimum
    return clusters

def modularity_based_clustering(graph:nx.Graph):
    return nx.greedy_modularity_communities(graph)