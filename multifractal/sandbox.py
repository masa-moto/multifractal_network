"""
Sandbox algorithm for multifractal analysis of complex networks.
ref: https://doi.org/10.1103/PhysRevE.103.043303

USAGE: from outside, call sandbox_analysis(graph, ...), get (q, tau(q), D(q)).

DESCRIPTION: calculate the mass of the subgraph within a given radius r from a randomly selected node.
    compute the average mass and its moments, and then estimate the generalized fractal dimensions D(q) via tau(q) (= (q-1)D(q)).
    in this implementation, we use breadth-first search (BFS) to find the nodes within distance r from the source node.

CONTENTS:this program file contains several sections:
- utility functions for calculation; linear_regression, diameter_approximation.
- core functions for sandbox algorithm;
    - preprocessing, init_worker functions
    - compute_sandbox_measure: compute the number of nodes within distance r from the source node
    - compute_sandbox: compute the normalized mass M(r)/N for a given source node
    - compute_Zq: compute Z(q) from the measure mu 
"""

import networkx as nx 
import numpy as np 
import scipy as cp 
import os
import csv
import random as rd 
import scipy.sparse.csgraph as ssc
import multiprocessing as mp

from numba import njit 
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from sklearn.linear_model import LinearRegression, TheilSenRegressor
from multiprocessing import cpu_count
from typing import List, Tuple, Dict, Sequence

#------------------- utility functions ------------------#

def linear_regression(x, y, method="theil-sen"):
    """
    Args:
        x (_type_): _description_
        y (_type_): _description_
        method (str, optional): Defaults to "ols". "ols"(ordinary least squares) and "theil-sen" are acceptable.
    Raises:
        ValueError: _description_

    Returns:
        _type_: _description_
    """
    x, y = np.array(x).reshape(-1, 1), np.array(y)
    # n = len(x)
    if method == "ols":
        model = LinearRegression().fit(x, y)
    elif method == "theil-sen":
        model = TheilSenRegressor(max_iter=int(1e3)).fit(x, y)
    else:
        raise ValueError(f"Unknown method {method}")
    return model.coef_[0], model.intercept_
  
def bfs_distance(graph, source):
    nodes = list(graph.nodes)
    csr = nx.to_scipy_sparse_array(graph, nodelist=nodes, format="csr")
    # unweighted=True で BFS、indices=source で start ノードのみ計算
    dist = ssc.shortest_path(
        csr,
        directed=False,
        unweighted=True,
        indices=source
    )
    # float→int にキャスト
    dist = dist.astype(int)
    # 辞書に詰め替え
    return dist

def _bfs_max_dist(candidate):
    return max(bfs_distance(_graph, candidate))

def diam_approx(graph, n_sample=10, scale_factor=1.05):
    def _init_worker_diam_approx(graph, nodes):
        global _graph, _nodes
        _graph, _nodes= graph, nodes
  
    nodes = list(graph.nodes)
    sample_nodes = np.random.choice(nodes, size = max(n_sample, int(.15*len(nodes))))
    max_length = []
    initargs = (graph, nodes)
    with ProcessPoolExecutor(max_workers=int(os.cpu_count()*.75), initializer=_init_worker_diam_approx, initargs=initargs) as executor:
        max_length = list(executor.map(_bfs_max_dist, sample_nodes))
    return int(max(max_length)*scale_factor)

def write_raw_measure(scale_mu:dict[int, np.ndarray], output_prefix:str) -> None:
    """
    Dumping scale -> mu values into \"*_rawmeasure.csv\" file.
    """
    base, _ = os.path.splitext(output_prefix)
    csv_file = f"{base}_rawmeasure.csv"
    max_len = max(map(len, scale_mu.values()))
    header = ["scale"] + [str(i) for i in range(max_len)]
    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for eps, mu in sorted(scale_mu.items()):
            row = [eps] + list(mu) + [""] * (max_len - len(mu))
            writer.writerow(row)

#------------------- preprocessing ------------------#
def process_graph(graph:nx.Graph, true_diameter=False) -> Tuple[int, int]:
    N = graph.number_of_nodes()
    if true_diameter:
        diam = nx.diameter(graph)
    else:
        diam = diam_approx(graph, n_sample=int(.15*N), scale_factor=1.0)
    return diam, N


#------------------- core functions ------------------#
def _init_worker_sandbox(graph_csr, nodes, scale, diam):
    global _graph_csr, _nodes, _scale, _diam
    _graph_csr, _nodes, _scale, _diam = graph_csr, nodes, scale, diam
    

def compute_sandbox_measure(graph_csr, N, source, scale)->Sequence[int]:
    order, pred = cp.sparse.csgraph.breadth_first_order(
        graph_csr, source, directed = False, return_predecessors = True
    )
    """
    Compute the number of nodes within distance r (0 ≤ r < diam)
    from the given source in the graph represented by graph_csr.
    Returns a tuple of length diam, where each element is the count of nodes at distance ≤ r.
    """
    dist = np.full(N, -1, dtype=int)
    dist[source] = 0
    for idx in order[1:]:
        if pred[idx] >= 0:
            dist[idx] = dist[pred[idx]] + 1
    return np.array([np.count_nonzero(dist<=r) for r in scale], dtype=float)#距離r以内のノード数

def compute_sandbox(graph_csr, scale, diam, source):
    # print(f"This is in sandbox.py/compute_sandbox(), args are follows,\n! ommit graph, scale:{scale}, diam:{diam}, source:{source}")
    N = graph_csr.shape[0]
    mu_array = compute_sandbox_measure(
                    graph_csr=graph_csr,
                    N=N,
                    source=source,
                    scale = scale
                )
    return mu_array/N

def _SB_measure_wrapper(source):
    """
    information of the graph is passed by global variable.
    hereafter, _graph_csr, _scale, _diam are passed by global variable.
    """
    return compute_sandbox(_graph_csr, _scale, _diam, source)#->np.array([M(r)/N for r in range(diam)])

@njit
def compute_Zq(mu:np.ndarray, q:float, method:str = "box_covering")->float:
    if len(mu) == 0:
        print('len(mu) = 0. please check again.')
        return 1.0
    elif method == "sandbox":
        # if q == 1.0:
        #     return -np.sum(mu * np.log(mu))
        return np.mean(mu ** (q-1))
    else:
        raise ValueError(f"[compute_Zq error] unknown method: {method}")
        

@njit
def compute_Zq_vectorized(scale_measures_array, valid_indices, q,):
    """
    NumPy配列とインデックスを使って高速にZqを計算
    """
    n_valid = len(valid_indices)
    results = np.empty(n_valid, dtype=np.float64)
    
    for i in range(n_valid):
        idx = valid_indices[i]
        mu = scale_measures_array[idx]
        
        # 0 < mu < 1 の要素のみを抽出（sandbox用）
        if len(mu) == 0:
            results[i] = 1.0
        else:
            results[i] = np.mean(mu ** (q-1))
        
    return results

def compute_Dq(
    tau: np.ndarray,
    q: np.ndarray,
    eps_thresh: float = 1e-5
)->np.ndarray:
    """
    compute Dq = tau(q)/(q-1), masking |q-1| <= eps_thresh as np.nan
    """
    denom = q - 1
    mask = np.abs(denom) > eps_thresh
    Dq = np.full_like(tau, np.nan, dtype=float)
    Dq[mask] = tau[mask] / denom[mask]
    return Dq
    
def calculate_mass_exponent(
    graph:nx.Graph,
    q_range:Tuple[float, float] = (-10, 10),
    q_ticks:float = 0.01,
    num_sandbox = 5000,
    scale_factor = 1.0,
    regression_method:str = "theil-sen",
    measure_output_prefix = None
) -> Tuple[np.ndarray, np.ndarray]:
    
    diam, N = process_graph(graph)
    print(f"[INFO] approx. diameter : {diam}")
    nodes = list(graph.nodes())
    q_array = np.linspace(min(q_range), max(q_range), int((max(q_range) - min(q_range))/q_ticks) + 1)
    num_sandbox = max(num_sandbox, int(0.15*N))
    if diam <= 3:
        scale = np.arange(3)
    else:
        scale = np.arange(2, int(diam*scale_factor))
    max_process = int(mp.cpu_count()**.9)
    print(f"[INFO] max process:{max_process}")
    q1, q3 = np.percentile(scale, [25, 75])
    iqr_msk = (q1 <= scale) & (scale <= q3)
    valid_scale = scale[iqr_msk]
    log_eps = np.log(valid_scale / diam)
    tau_q = []
    log_Zs_q = []
    
    graph_csr = nx.to_scipy_sparse_array(graph, nodelist=nodes, format = "csr")
    source_candidates = np.random.choice(nodes, size = num_sandbox)
    initargs = (graph_csr, nodes, scale, diam)
    with mp.Pool(max_process, initializer=_init_worker_sandbox, initargs=initargs) as pool:
        result = np.array(pool.map(_SB_measure_wrapper, source_candidates))
    # 辞書の代わりに配列インデックスベースの高速アクセスを使用
    result_T = result.T
    # scaleの最小値でオフセットして配列インデックスに変換
    scale_offset = scale[0]  # scale = np.arange(2, diam*8) なので最小値は2
    scale_measures = result_T  # shape: (len(scale), num_sandbox)

    if measure_output_prefix:
        scale_mu_dict = dict(zip(scale, scale_measures))
        write_raw_measure(scale_mu_dict, measure_output_prefix)

    valid_indices = valid_scale - scale_offset

    for q in q_array:
        Zs = compute_Zq_vectorized(scale_measures, valid_indices, q)
        log_Zs = Zs if q == 1.0 else np.log(Zs)
        assert np.all(np.isfinite(log_Zs)), f"[calculate_mass_exponent error] invalid value encountered. possible inf, nan. {log_Zs}"
        log_Zs_q.append(log_Zs)
        
    # print(np.isfinite(log_Zs_q).all())  # → False なら問題箇所あり
    for i, q in enumerate(q_array):
        assert np.all(np.isfinite(log_Zs_q[i])), f"invalid value encountered. possible inf, nan. {log_Zs_q[i]} "
        
        slope, _ = linear_regression(log_eps, log_Zs_q[i], method = regression_method)
        tau_q.append(slope)
    return q_array, np.array(tau_q)


#------------------- wrapper ------------------#
def sandbox_analysis(
    graph:nx.Graph,
    q_range:Tuple[float, float] = (-10, 10),
    q_ticks:float = 0.01,
    num_sandbox = 5000,
    scale_factor = 1.0,
    regression_method:str = "theil-sen",
    measure_output_prefix = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Wrapper function for sandbox algorithm.
    returns (q_array, tau(q), D(q))
    """
    q_array, tau_array = calculate_mass_exponent(
        graph=graph,
        q_range=q_range,
        q_ticks=q_ticks,
        num_sandbox=num_sandbox,
        scale_factor=scale_factor,
        regression_method=regression_method,
        measure_output_prefix=measure_output_prefix
    )
    Dq_array = compute_Dq(tau_array, q_array)
    return q_array, tau_array, Dq_array

#------------------- main ------------------#

if __name__ == "__main__":
    nodes = 5000
    m = 3
    print(f"[INFO] sandbox analysis for Barabasi Albert scale-free network with {nodes} nodes, m={m}.")

    g = nx.barabasi_albert_graph(nodes, m, seed=0, initial_graph=nx.complete_graph(4))
    print(f"[INFO] graph generated. n={g.number_of_nodes()}, m={g.number_of_edges()}")
    q, tau, Dq = sandbox_analysis(
        g,
        q_range=(-10, 10),
        q_ticks=0.1,
        num_sandbox=500,
        scale_factor=1.0,
        regression_method="theil-sen",
        measure_output_prefix="test_sandbox"
    )
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    axes[0].plot(q, Dq, markersize=4, color="blue")
    axes[0].set_xlabel("q")
    axes[0].set_ylabel("D(q)")
    axes[0].set_title("D(q) vs q")
    axes[0].grid()

    axes[1].plot(q, tau, marker="o", color="orange")
    axes[1].set_xlabel("q")
    axes[1].set_ylabel("tau(q)")
    axes[1].set_title("tau(q) vs q")
    axes[1].grid()
    fig.savefig("Dq_vs_q.png")
    print(f"[INFO] figure saved as Dq_vs_q.png")