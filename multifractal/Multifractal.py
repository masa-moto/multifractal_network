import networkx as nx
import numpy as np
import multiprocessing as mp
import random as rand
import csv
import os 
from typing import Tuple, List, Union
from numba import njit
from .core import reg1dim, process_graph, write_raw_measure
from .sandbox import compute_sandbox
from . import box_covering as _bc

_graph_csr, _nodes, _scale, _diam = None, None, None, None


def _BC_measure_wrapper(args: Tuple[nx.Graph, int, int]) -> Tuple[int, np.ndarray]:
    graph, eps, diam = args
    mu = np.array(_bc.multifractal_box_covering(graph, eps, diam=diam, N=graph.number_of_nodes()))
    return eps, mu[(0 < mu) & (mu < 1)]

def _SB_measure_wrapper(source):
    """
    graphについての情報はinitializerでglobal variableとして受け取っていることに注意。
    以下、_hogehogeについても同様にglobalとして受け取っているらしい
    """
    return compute_sandbox(_graph_csr, _scale, _diam, source)#->np.array([M(r)/N for r in range(diam)])

@njit
def compute_Zq(mu:np.ndarray, q:float, method:str = "box_covering")->float:
    if len(mu) == 0:
        print('len(mu) = 0. please check again.')
        return 1
    elif method == "box_covering":        
        if q == 1.0:
            return -np.sum(mu * np.log(mu))
        return np.sum(mu ** q)    
    elif method == "sandbox":
        # if q == 1.0:
        #     return -np.sum(mu * np.log(mu))
        return np.mean(mu ** (q-1))
    else:
        raise ValueError(f"[compute_Zq error] unknown method: {method}")

@njit
def compute_Zq_vectorized(scale_measures_array, valid_indices, q, method="sandbox"):
    """
    NumPy配列とインデックスを使って高速にZqを計算
    """
    n_valid = len(valid_indices)
    results = np.empty(n_valid, dtype=np.float64)
    
    for i in range(n_valid):
        idx = valid_indices[i]
        mu = scale_measures_array[idx]
        
        # 0 < mu < 1 の要素のみを抽出（sandbox用）
        if method == "sandbox":
            if len(mu) == 0:
                results[i] = 1.0
            else:
                results[i] = np.mean(mu ** (q-1))
        else:  # box_covering
            valid_mu = mu[(mu > 0) & (mu < 1)]
            if len(valid_mu) == 0:
                results[i] = 1.0
            elif q == 1.0:
                results[i] = -np.sum(valid_mu * np.log(valid_mu))
            else:
                results[i] = np.sum(valid_mu ** q)
    
    return results

def _init_worker_sandbox(graph_csr, nodes, scale, diam):
    global _graph_csr, _nodes, _scale, _diam
    _graph_csr, _nodes, _scale, _diam = graph_csr, nodes, scale, diam
    
def calculate_mass_exponent(
    graph:nx.Graph,
    q_range:Tuple[float, float] = (-10, 10),
    q_ticks:float = 0.01,
    method:str = "sandbox",
    num_sandbox = 5000,
    scale_factor = 1.0,
    regression_method:str = "theil-sen",
    measure_output_prefix = None
) -> Tuple[np.ndarray, np.ndarray]:
    #input and preparation---#
    diam, N = process_graph(graph)
    print(f"approx. diameter : {diam}")
    nodes = list(graph.nodes())
    q_array = np.linspace(min(q_range), max(q_range), int((max(q_range) - min(q_range))/q_ticks) + 1)
    num_sandbox = max(num_sandbox, int(0.15*N))
    if diam <= 3:
        scale = np.arange(3)
    else:
        scale = np.arange(2, int(diam*scale_factor))
        
    #process branching---#
    #   |-A: classical coverings
    #   |-B: sandbox coverings
    #   -> merge at regression process
    
    #branch A: classical covering#
    max_process = int(mp.cpu_count()**.9)
    print(f"max process:{max_process}")
    q1, q3 = np.percentile(scale, [25, 75])
    iqr_msk = (q1 <= scale) & (scale <= q3)
    valid_scale = scale[iqr_msk]
    log_eps = np.log(valid_scale / diam)
    tau_q = []
    log_Zs_q = []
    if method == "box_covering":
        with mp.Pool(max_process) as pool:
            result = pool.map(_BC_measure_wrapper, [(graph, eps, diam) for eps in scale])
        scale_mu = {eps: mu for eps, mu in result} #key=eps (covering radius), item=mu (sequence of raw measure)

    #branch B: sandbox covering#
    elif method == "sandbox":
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
        
    else:
        raise ValueError(f'Unknown method:{method}.')
    
    if measure_output_prefix:
        scale_mu_dict = dict(zip(scale, scale_measures))
        write_raw_measure(scale_mu_dict, measure_output_prefix)
        # base, _ = os.path.splitext(measure_output_prefix)
        # csv_file = f"{base}_rawmeasure.csv"
        # # print(f"measure_output_prefix: {measure_output_prefix}")
        # max_len = max(len(v) for v in scale_mu.values())
        # header = ["scale"] + [str(i) for i in range(max_len)]
        # with open(csv_file, "w", newline = "") as f:
        #     writer = csv.writer(f)
        #     writer.writerow(header)
        #     for scale, measure in scale_mu.items():
        #         row = [scale] + list(measure) + [""]*(max_len - len(measure))
        #         writer.writerow(row)

    #merge branch#

    valid_indices = valid_scale - scale_offset

    for q in q_array:
        Zs = compute_Zq_vectorized(scale_measures, valid_indices, q, method)
        log_Zs = Zs if q == 1.0 else np.log(Zs)
        assert np.all(np.isfinite(log_Zs)), f"[calculate_mass_exponent error] invalid value encountered. possible inf, nan. {log_Zs}"
        log_Zs_q.append(log_Zs)
        
    # print(np.isfinite(log_Zs_q).all())  # → False なら問題箇所あり
    for i, q in enumerate(q_array):
        assert np.all(np.isfinite(log_Zs_q[i])), f"invalid value encountered. possible inf, nan. {log_Zs_q[i]} "
        
        slope, _ = reg1dim(log_eps, log_Zs_q[i], method = regression_method)
        tau_q.append(slope)
    return q_array, np.array(tau_q)
