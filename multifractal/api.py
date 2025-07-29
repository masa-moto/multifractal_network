import numpy as np 
from typing import Dict, Tuple, List
import csv
from .Multifractal import calculate_mass_exponent
from .core import reg1dim


    
def mass_exponent(graph, qrange, q_ticks=0.1, method = "sandbox", raw_measure_output_path = None, raw_measure_input_path=None):
    """_summary_

    Args:
        graph (nx.Graph): _description_
        qrange (np.ndarray): _description_

    Returns:
        q_array (np.ndarray):
        tau_array (np.ndarray): 
    """
    if raw_measure_input_path:
        return _q_powered_mean(raw_measure_input_path, qrange)
    else:
        return calculate_mass_exponent(graph, qrange, q_ticks = q_ticks,  method=method, measure_output_prefix=raw_measure_output_path)

def _q_powered_mean(rawmeasure_input_path, qrange):
    data = np.loadtxt(rawmeasure_input_path, delimiter=",", skiprows = 1)
    if data.shape[0]<2:
        raise ValueError(f"There is no data row.")
    q_array = np.linspace(min(qrange), max(qrange), int(1e3))
    r_array = list(range(1, len(data)))
    slope_list = []
    for q in q_array:
        powered= [np.power(data[i], q) for i in range(len(data))]
        
def recalculate_tauq(data_path, q_range, q_nums):
    q_array = np.linspace(min(q_range), max(q_range), q_nums)
    with open(data_path) as f:
        reader = csv.reader(f)
        header = next(reader)
        data = np.array([list(map(float,row)) for row in reader])
    r_array = [i for i in range(len(data))]
    slopes = []
    for q in q_array:
        powered = np.array([np.mean(datum[1:]**q) for datum in data])
        slopes.append(reg1dim(r_array, powered)[0])
    return q_array, slopes

def recalc_tauq_fast(
    data_path: str,
    q_range: tuple[float, float],
    q_nums: int = 1000,
    method: str = "sandbox"
) -> tuple[np.ndarray, np.ndarray]:
    """
    raw_measure CSV を読み込んで τ(q) を再計算する高速版。
    - data_path: scale + measures... の CSV
    - q_range: (q_min, q_max)
    - q_nums: q の分割数
    - method: "sandbox" or "box_covering"
    """
    # 1) ファイル読み込み
    arr = np.loadtxt(data_path, delimiter=",", skiprows=1)
    scales = arr[:, 0]
    mu_all = arr[:, 1:]  # shape (S, T)

    # 2) q 軸配列，スケールログ
    q_array = np.linspace(q_range[0], q_range[1], q_nums)
    valid = scales > 0
    x = np.log(scales[valid])
    x0 = x - x.mean()
    denom = np.sum(x0 * x0)

    # 3) μ**q をまとめて計算 → Z(q, r)
    #    sandbox: ⟨μ^(q-1)⟩, box_covering: ∑μ^q
    mu = mu_all[valid]      # (S, T)
    Q, S, T = q_array.size, mu.shape[0], mu.shape[1]
    # broadcast で (Q, S, T) を作る
    if method == "sandbox":
        P = q_array[:, None, None] - 1
        Z = np.mean(mu[None, :, :] ** P, axis=2)   # → (Q, S)
    else:
        P = q_array[:, None, None]
        Z = np.sum(mu[None, :, :] ** P, axis=2)    # → (Q, S)

    # 4) log を取って回帰 slope をまとめて計算
    #    q=1 特殊扱いしたい場合はここで Z[qi] を -sum μ log μ に差し替え
    Y = np.log(Z)
    Ym = Y.mean(axis=1, keepdims=True)           # (Q,1)
    cov = np.sum((Y - Ym) * x0[None, :], axis=1)  # (Q,)
    tau = cov / denom

    return q_array, tau
