import os
import csv
import random as rd 
from typing import Tuple
import scipy.sparse.csgraph as ssc
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

import networkx as nx 
import numpy as np 
from sklearn.linear_model import LinearRegression, TheilSenRegressor
_graph, _nodes = None, None

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

def _init_worker_diam_approx(graph, nodes):
    global _graph, _nodes
    _graph, _nodes= graph, nodes
    
def diam_approx(graph, n_sample=10, scale_factor=1.05):
    nodes = list(graph.nodes)
    sample_nodes = np.random.choice(nodes, size = max(n_sample, int(.15*len(nodes))))
    max_length = []
    initargs = (graph, nodes)
    with ProcessPoolExecutor(max_workers=int(os.cpu_count()*.75), initializer=_init_worker_diamapprox, initargs=initargs) as executor:
        max_length = list(executor.map(_bfs_max_dist, sample_nodes))
    return int(max(max_length)*scale_factor)

def reg1dim(x, y, method="theil-sen"):
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
    # a = float(((np.dot(x, y)- y.sum() * x.sum()/n)/((x ** 2).sum() - x.sum()**2 / n)))
    # b = (y.sum() - a * x.sum())/n
    # return a, b


def process_graph(graph:nx.Graph, true_diameter=False) -> Tuple[int, int]:
    N = graph.number_of_nodes()
    if true_diameter:
        diam = nx.diameter(graph)
    else:
        diam = diam_approx(graph, n_sample=int(.15*N), scale_factor=1.0)
    return diam, N


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
    
def compute_Dq(
    tau: np.ndarray,
    q: np.ndarray,
    eps_thresh: float = 1e-4
)->np.ndarray:
    """
    compute Dq = tau(q)/(q-1), masking |q-1| <= eps_thresh as np.nan
    """
    denom = q - 1
    mask = np.abs(denom) > eps_thresh
    Dq = np.full_like(tau, np.nan, dtype=float)
    Dq[mask] = tau[mask] / denom[mask]
    return Dq