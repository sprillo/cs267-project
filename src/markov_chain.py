import numpy as np


def compute_stationary_distribution(
    rate_matrix: np.array
) -> np.array:
    eigvals, eigvecs = np.linalg.eig(rate_matrix.transpose())
    eigvals = eigvals.real
    eigvecs = eigvecs.real
    eigvals = np.abs(eigvals)
    index = np.argmin(eigvals)
    stationery_dist = eigvecs[:, index]
    stationery_dist = stationery_dist / sum(stationery_dist)
    return stationery_dist
