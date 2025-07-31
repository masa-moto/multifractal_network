import networkx as nx 
import numpy as np 
import matplotlib.pyplot as plt 
import random as rd 
from clustering import entropy_based_clustering
import random

def random_color(seed=None):
    if seed is not None:
        random.seed(seed)
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    return f'#{r:02X}{g:02X}{b:02X}'

if __name__ == "__main__":
    g = nx.barabasi_albert_graph(1000, 1, 0)

    clusters = entropy_based_clustering(g)
    validation_clst = set()
    for clst in clusters:
        idx, cluster = clst
        validation_clst |= cluster
        print(len(cluster), end = "  ")
    
    outlanders = set(g.nodes()) - validation_clst
    print(outlanders)
    fig = plt.figure()
    ax = fig.add_subplot()
    pos = nx.nx_agraph.graphviz_layout(g, prog = "sfdp")
    nx.draw_networkx_edges(g, pos=pos, ax = ax, label = None)
    for clust in clusters:
        idx, cluster = clust
        nx.draw_networkx(
            g, pos, ax = ax,
            nodelist=list(cluster),
            node_color = random_color(idx),
            node_size = 25,
            with_labels= False
            )
    nx.draw_networkx_nodes(g, pos, outlanders, node_color="r", node_size = 15)
    fig.savefig("test.png")