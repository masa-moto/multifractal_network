"""
k-cluster filtration for TDA of complex networks.
ref: https://doi.org/10.3390/e25121587

USAGE: from outside, call k_cluster_filtration(grpah, ...), get ().
graph should be weighted, otherwise random values between [0, 1] by uniform distribution is assigned as the weight of each edges.
# sandbox_analysis(graph, ...), get (q, tau(q), D(q)).

DESCRIPTION: 
calculate the mass of the subgraph within a given radius r from a randomly selected node.
    compute the average mass and its moments, and then estimate the generalized fractal dimensions D(q) via tau(q) (= (q-1)D(q)).
    in this implementation, we use breadth-first search (BFS) to find the nodes within distance r from the source node.

CONTENTS:this program file contains several sections;
- utility functions for calculation; linear_regression, diameter_approximation.
- core functions for sandbox algorithm;
    - preprocessing, init_worker functions
    - compute_sandbox_measure: compute the number of nodes within distance r from the source node
    - compute_sandbox: compute the normalized mass M(r)/N for a given source node
    - compute_Zq: compute Z(q) from the measure mu 
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
                
                if self.size[new_root] >= self.k and new_root not in self.persistence:
                    self.persistence[new_root] = w # store information as birth time
                if ru in self.persistence and self.size[ru] < self.k:
                    self.lifetimes.append(ClusterLifetime(self.persistence[ru], w, self.size[ru]))
                    del self.persistence[ru]
                    print(f"dead{ru}")
                    
                if rv in self.persistence and self.size[rv] < self.k:
                    self.lifetimes.append(ClusterLifetime(self.persistence[rv], w, self.size[rv]))
                    del self.persistence[rv]
                    print(f"dead{rv}")
        for root, birth in self.persistence.items():
            self.lifetimes.append(ClusterLifetime(birth, float("inf"), self.size[root]))
        return self.lifetimes
    
    
    

if __name__ == "__main__":
    g = nx.barabasi_albert_graph(200, 1, 0, nx.complete_graph(4))
    filter = k_cluster_filtration(g, k=5, seed = 0)
    for ff in filter.fit():
        print(ff.duration())