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
    graph = _share_graph
    cluster, seed = args
    
    # initial cluster processing...
    cluster = core.update_cluster(graph, set(cluster), seed, "internal")
    
    # final cluster processing...
    return seed, core.update_cluster(graph, cluster, seed, "boundary")

def entropy_based_clustering(graph:nx.Graph, threshold:float = 1e-10) -> List[FrozenSet]:
    """
    find GE(graph entropy) based clustering. this function uses algorithm by C.K.Edward and C.Young-Rae (doi:10.1109/ICDM.2011.64)

    Args:
        grpah (nx.Graph): graph to be examined.

    Returns:
        List[FrozenSet]: index of return represents seed nodes and FrozenSet represents GE based cluster of corresponding seed node.
    """
    
    nodes = graph.nodes()
    # create initial cluster consisted of seed and its neighbours
    init_clusters=[list(graph.neighbors(n)) for n in nodes]
    # see if there is any candidate to delete from cluster to minimize GE
    init_args = (graph, "dum")
    args = zip(init_clusters, nodes)
    with mp.Pool(processes= 1, initializer=graph_initializer, initargs=init_args) as p:
        clusters = p.map(_update_cluster_wrapper, args)
    # calculate GE and update cluster until GE becomes minimum
    return clusters

def modularity_based_clustering(graph:nx.Graph):
    return nx.greedy_modularity_communities(graph)