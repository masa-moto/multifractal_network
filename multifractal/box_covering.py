import networkx as nx 
import numpy as np 
import numba as nb
import random as rand
import multiprocessing as mp 
from typing import Tuple, Union, List

def box_covering(
    G: nx.Graph,
    radius: int,
    seed_selector_name: str ="random",
    seed_from_all:bool = False
    ) -> Tuple[Union[int, List[float]]]:
    
    all_nodes = set(G.nodes)
    N=len(all_nodes)
    covered_nodes = set()
    def _random_seed_all(G, covered_nodes, all_nodes):
        return rand.choice(list(all_nodes))
    
    def _random_seed_uncovered(G, covered_nodes, all_nodes):
        return rand.choice(list(all_nodes - covered_nodes))
    
    def _max_degree_seed(G, covered_nodes, all_nodes):
        candidates = list(all_nodes - covered_nodes)
        return     max(candidates, key = lambda node: G.degree[node])
    
    def _min_degree_seed(G, covered_nodes, all_nodes):
        candidates = list(all_nodes - covered_nodes)
        return     min(candidates, key = lambda node: G.degree[node])
    
    seed_selector_dict = {
        "all": _random_seed_all,
        "random": _random_seed_uncovered,
        "maxdeg": _max_degree_seed,
        "mindeg": _min_degree_seed
    }
    
    if seed_selector_name not in seed_selector_dict:
       raise ValueError(f'Unknown seed selector name: {seed_selector_name}')
    
    seed_selector = seed_selector_dict[seed_selector_name]
    measure_list = []
    
    while covered_nodes != all_nodes:
        seed = seed_selector(G, covered_nodes, all_nodes)
        box_nodes = set(nx.single_source_shortest_path_length(G, seed, cutoff=radius).keys())
        new_nodes = box_nodes - covered_nodes
        if new_nodes:
            measure_list.append(len(new_nodes)/N)
            covered_nodes.update(new_nodes)
    return radius, measure_list

def multifractal_box_covering(
    graph:nx.Graph,
    box_radii:int,
    method= "covering",
    covering_trial = 100,
    dist_mtx = np.array(list(range(2))),
    diam = 1,
    N = 2,
    seed_selector_name = "maxdeg"
    ) -> List[float]:
    
    if not nx.is_connected(graph):
        raise ValueError('Graph should be connected.')
     
    uncovered_node = set(graph.nodes())
    # dist_mtx, diam, N = process_graph(graph)
    
    if method == "Csong":
        measure_list = []
        while uncovered_node:
            seed = rand.choice(tuple(uncovered_node))
            box = {j for j in uncovered_node if dist_mtx[seed][j] <= box_radii}
            measure_list.append(len(box)/N)
            uncovered_node -= box
    elif method == "covering":
        measure_list = []
        for _ in range(covering_trial):
            _, result = box_covering(graph, box_radii, seed_selector_name=seed_selector_name)
            if not measure_list or len(result) < len(measure_list):
                measure_list = result
    else:
        raise ValueError(f'Unknown method:{method}.')
                        
    return measure_list