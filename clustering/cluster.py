# main program file for clustering and its around.
# GE(graph entropy) or modularity based clustering functions, and visualization function.
import networkx as nx 
import numpy as np 
import multiprocessing as mp 
import matplotlib.pyplot as plt 
from typing import FrozenSet, List, Set, Dict

from .core import update_cluster, seed_sorter


_share_graph = nx.Graph()
_share_threshold = 1e-10
_deg_dict=None
def graph_initializer(graph:nx.Graph, threshold:float, deg_dict: Dict):
    global _share_graph
    global _share_threshold
    global _deg_dict
    _deg_dict = deg_dict
    _share_graph = graph
    _share_threshold = threshold


def _update_cluster_wrapper(args):
    
    cluster, seed = args
    # initial cluster processing...
    cluster = update_cluster(_share_graph, set(cluster), seed, "internal", cutoff = _share_threshold, deg_dict=_deg_dict)
    # final cluster processing...

    
    return seed, update_cluster(_share_graph, cluster, seed, "boundary", cutoff = _share_threshold, deg_dict=_deg_dict)

def entropy_based_clustering(
    graph:nx.Graph,
    cluster_cutoff_size:int = 2,
    GE_threshold:float = 1e-3,
    ) -> List[Set]:
    """
    find GE(graph entropy) based clustering. this function uses algorithm by C.K.Edward and C.Young-Rae (doi:10.1109/ICDM.2011.64)

    Args:
        grpah (nx.Graph): graph to be examined.
        cluster_cutoff_size(int): minimum number of nodes of cluster.
        GE_threshold(float): threshold for updating GE. when difference of GE becomes below threshold, algorithm halts cluster update.

    Returns:
        List[Tuple[int, Set]]: index of return represents seed nodes and Set represents GE based cluster of corresponding seed node.
    """
    
    

    # create initial cluster consisted of seed and its neighbours
    nodes = graph.nodes
    graph_csr = nx.to_scipy_sparse_array(graph, nodelist=nodes, format = "csr")
    init_clusters = [set(nx.neighbors(graph, node)) | {node} for node in nodes]#[set(map(int, graph_csr[n:n+1,:].nonzero()[1])) for n in range(graph_csr.shape[0])]
    deg_dict = dict(graph.degree)
    # see if there is any candidate to delete from cluster to minimize GE
    init_args = (graph, GE_threshold, deg_dict)
    args = zip(init_clusters, nodes)
    with mp.Pool(processes= 1, initializer=graph_initializer, initargs=init_args) as p:

        clusters = p.map(_update_cluster_wrapper, args)
    # filtering cluster by their size
    clusters = [(seed, cluster) for seed, cluster in clusters if len(cluster) > max(1, cluster_cutoff_size)]
    return clusters

def modularity_based_clustering(graph:nx.Graph, weight=None, resolution=1, cutoff=1, best_n=None):
    return nx.community.greedy_modularity_communities(graph, weight, resolution, cutoff, best_n)

def draw_clusters(graph, pos, clusters, fig_path, num_cluster = 5, order = "descending"):
    """
    function for drawing graph mainly focusing on cluster structure.
    highlight a cluster in the graph and lists the specified number(num_cluster) of cluster in specified order (order).
    any clusters consisting of exactly the same nodes are treated as the same cluster.

    Args:
        graph (nx.Graph): graph to be drawn
        pos (dict): positions of all components of graph.
        clusters (dict): dictionary of clusters whose items are {seed_node : set([incluster nodes])}.
        fig_path (str): path for saved figure.
        num_cluster (int, optional): number of cluster to be drawn. Defaults to 5. if num_cluster=-1, all the clusters are drawn.
        order (str or iterable,  optional): cluster selection order. Defaults to "descending". str input "descending", "ascending" are supported.
        If there is user input of order of seeds, that order is selected.
    """
    # specifying unique clusters
    unique_clusters = {}
    cluster_groups = {}
    seeds = seed_sorter(None, clusters)
    for seed in seeds:
        cluster_set = set(clusters[seed])
        matched_seed = None
        for u_seed, u_cluster in unique_clusters.items():
            if cluster_set == u_cluster:
                matched_seed = u_seed
                break

        if matched_seed is None:
            # unique cluster detected:
            unique_clusters[seed] = cluster_set
            cluster_groups[seed] = [seed]
        else:
            # known cluster detected:
            cluster_groups[matched_seed].append(seed)
    # updating num_cluster when num_cluster < unique clusters count.
    num_cluster = min(num_cluster, len(unique_clusters.keys()))
    
    # setting position if None
    if pos is None:
        pos = nx.nx_agraph.pygraphviz_layout(graph, prog="sfdp")
        
    # ordering seed nodes    
    seeds = seed_sorter(None, unique_clusters)
    
    #specifying cluster to be drawn. all or specified
    if num_cluster == -1:
        num_cluster = len(unique_clusters)
    else:
        num_cluster = min(num_cluster, len(unique_clusters.keys()))
    if num_cluster:
        seeds = seeds[:num_cluster]
    
    # plotting section.
    fix, axes = plt.subplots(num_cluster, 2, figsize = (12, 4*num_cluster))
    for ax, seed in zip(axes, seeds):
        cluster_nodes = unique_clusters[seed]
        node_color = ["b" if n == seed else "r" if n in cluster_nodes else "lightgray" for n in graph.nodes()]
        edge_color = ["k" if u in cluster_nodes and v in cluster_nodes else "lightgray" for u, v in graph.edges()]
        
        #left side plot: cluster in entire graph
        nx.draw_networkx(
            graph, 
            pos,
            node_color=node_color,
            edge_color=edge_color,
            with_labels=True,
            node_size =50,
            ax = ax[0])
        ax[0].set_title(f"full graph - cluster: seed {seed}. size = {len(cluster_nodes)}")
        
        #right side plot: cluster structure
        subgraph = graph.subgraph(unique_clusters[seed]).copy()
        node_color = ["b" if n == seed else "r" for n in subgraph.nodes()]
        nx.draw_networkx(
            subgraph,
            pos = nx.kamada_kawai_layout(subgraph),
            ax = ax[1],
            node_color = node_color
        )
        ax[1].set_title(f"cluster seed {cluster_groups[seed]}. size = {len(cluster_nodes)}")
    
    plt.tight_layout()
    plt.savefig(fig_path)
    