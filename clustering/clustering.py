import networkx as nx 
import numpy as np 
import multiprocessing as mp 
from typing import FrozenSet, List

import entropy


def entropy_based_clustering(graph:nx.Graph, threshold = 1e-10) -> List[FrozenSet]:
    """
    find GE(graph entropy) based clustering. this function uses algorithm by C.K.Edward and C.Young-Rae (doi:10.1109/ICDM.2011.64)

    Args:
        grpah (nx.Graph): graph to be examined.

    Returns:
        List[FrozenSet]: index of return represents seed nodes and FrozenSet represents GE based cluster of corresponding seed node.
    """
    nodes = graph.nodes()
    # create initial cluster consisted of seed and its neighbours
    cluster=[set(graph[n]) for n in nodes]
    # see if there is any candidate to delete from cluster to minimize GE
    
    # calculate GE and update cluster until GE becomes minimum
    
    
    clusters = frozenset()
    
    return clusters

def modularity_based_clustering(graph:nx.Graph):
    return nx.greedy_modularity_communities(graph)