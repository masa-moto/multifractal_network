import numpy as np 
from numba import njit 


@njit
def power_iteration(A, num_iter=1000, cutoff=1e-10):
    n, _ = A.shape
    b = np.random.rand(n)
    b = b / np.linalg.norm(b)

    for _ in range(num_iter):
        Ab = A @ b
        norm_Ab = np.linalg.norm(Ab)
        if norm_Ab < cutoff:
            break
        b_next = Ab / norm_Ab
        if np.linalg.norm(b - b_next) < cutoff:
            break
        b = b_next

    # 近似固有値（Rayleigh quotient）
    eigenvalue = b @ (A @ b)
    return eigenvalue, b  # 最大固有値、固有ベクトル