"""
k-cluster filtration for TDA of complex networks.
ref: https://doi.org/10.3390/e25121587

USAGE: from outside, call k_cluster_filtration(grpah, ...), get ().
graph should be weighted, otherwise random values between [0, 1] by uniform distribution is assigned as the weight of each edges.
# sandbox_analysis(graph, ...), get (q, tau(q), D(q)).

DESCRIPTION: k-cluster fitration is a method to extract persistent clusters from weighted networks.
    the method is based on the union-find algorithm, and it tracks the birth and death of clusters as edges are added in order of their weights.
    the output is a list of ClusterLifetime objects, each containing the birth time, death time, and size of a cluster.
    if a cluster does not die (i.e., it persists until the end of the filtration), its death time is recorded as infinity.
    the duration of a cluster is defined as the difference between its death and birth times, with infinity if it does not die.
    this implementation also includes optimizations such as path compression and union by size.


CONTENTS:this program file contains several sections;
- utility;
    - dataclass ClusterLifetime: to store information of each cluster. birth, death, size are stored as attributes. duration() method returns the duration of the cluster.
- core;
    - class k_cluster_filtration: main class for k-cluster filtration.
    - wrapper function 
"""
#-------------- imports  ---------------#
import networkx as nx 
import random as rd
from typing import Iterable
from dataclasses import dataclass
#-------------- utilities --------------#
@dataclass 
class ClusterLifetime:
    birth: float
    death: float
    size: int
    def duration(self):
        if self.death == float("inf"):
            return self.death
        return self.death - self.birth

#---------------- core -----------------#
class k_cluster_filtration():
    def __init__(self, graph, k, seed = None, weight_attr = "weight"):
        self.graph = graph
        self.k = k
        self.seed = seed
        self.weight_attr = weight_attr
        self.parent = {v:v for v in self.graph.nodes()}
        self.size = {v:1 for v in self.graph.nodes()}
        self.persistence = {}
        self.lifetimes = []
        
        if not seed:
            print(f"[info] seed setting for random variable:{self.seed}")
            rd.seed(self.seed)
        
        
        if not nx.is_weighted(self.graph, weight=self.weight_attr):
            print("[warn] graph is not weighted. random weights are assigned to edge")
            for e in self.graph.edges(data = True):
                e[2]["weight"] = rd.uniform(0, 1)
        self.edge_sorted = sorted(self.graph.edges(data=True), key = lambda x:x[2].get("weight", 1.0))
        
        
    def find(self, v):#search and return canonical representative of v (the parent of parents with regard to v)
        while self.parent[v] != v:
            """swapしながら系統樹の階層を登っていくイメージ。系統樹の頂点(parent of parents)は自身をparentに持つ。"""
            self.parent[v] = self.parent[self.parent[v]]
            v = self.parent[v]
        return v
    
    def merge(self, u, v): #merge two components containing u/v into one components, including updating root(referred as the parent of parents)
        ru, rv = self.find(u), self.find(v) #the canonical representative for connected component containing u/v
        if ru == rv:
            return ru, False
        if self.size[ru]<self.size[rv]: # the size of ru should be larger than that of rv
            ru, rv = rv, ru
        self.parent[rv] = ru    #set ru as the parent of rv
        self.size[ru] += self.size[rv]
        return ru, True
    
    def fit(self):
        self.lifetimes.clear()
        for u, v, data in self.edge_sorted:
            w = data.get("weight", 1.0)
            ru, rv = self.find(u), self.find(v)
            if ru != rv:
                new_root, merged = self.merge(u, v)
                dead_root = rv if new_root == ru else ru
                if self.size[new_root] >= self.k and new_root not in self.persistence:
                    self.persistence[new_root] = w # store information as birth time

                if dead_root in self.persistence:
                    self.lifetimes.append(ClusterLifetime(self.persistence[dead_root], w, self.size[dead_root]))
                    del self.persistence[dead_root]
                # if ru in self.persistence and merged == True:
                #     self.lifetimes.append(ClusterLifetime(self.persistence[ru], w, self.size[ru]))
                #     del self.persistence[ru]
                #     print(f"dead{ru}")
                    
                # if rv in self.persistence and merged == True:
                #     self.lifetimes.append(ClusterLifetime(self.persistence[rv], w, self.size[rv]))
                #     del self.persistence[rv]
                #     print(f"dead{rv}")
        for root, birth in self.persistence.items():
            self.lifetimes.append(ClusterLifetime(birth, float("inf"), self.size[root]))
        return self.lifetimes
    
    
    

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import heapq
    def plot_PD(lifetimes):
        births = [c.birth for c in lifetimes]
        deaths = [c.death/c.birth for c in lifetimes]
        _, second = heapq.nlargest(2, deaths)
        plt.figure(figsize=(5,5))
        for b, d in zip(births, deaths):
            if d == float("inf"):
                long_cluster_b = b
                plt.scatter(b, (max(births)+second)*1.1, marker="^", c="k", zorder=5,)  # 無限寿命は三角で上に描く
            else:
                plt.scatter(b, d, c="b")
        lim = max(births + [d for d in deaths if d != float("inf")])
        plt.plot([0, max(births)], [ (max(births)+second)*1.1, (max(births)+second)*1.1], "r--", zorder=1, label=f"surviving cluster\nbirth: {long_cluster_b:.4f}")  # 反対側の対角線
        # plt.plot([0, lim], [0, lim], "k--")  # 対角線
        plt.xlabel("Birth")
        plt.ylabel("Death/Birth")
        plt.legend(loc="upper right")
        plt.title("Persistence Diagram")
        plt.savefig("persistence_diagram.png")

    g = nx.barabasi_albert_graph(10000, 2, 0, nx.complete_graph(4))
    filter = k_cluster_filtration(g, k=5, seed = 0)
    for ff in filter.fit():
        print(ff)
    print(filter.persistence)
    plot_PD(filter.lifetimes)