import os

import numpy as np
import pandas as pd
import torch


def compute_stationary_distribution(rate_matrix: np.array) -> np.array:
    eigvals, eigvecs = np.linalg.eig(rate_matrix.transpose())
    eigvals = eigvals.real
    eigvecs = eigvecs.real
    eigvals = np.abs(eigvals)
    index = np.argmin(eigvals)
    stationary_dist = eigvecs[:, index]
    stationary_dist = stationary_dist / sum(stationary_dist)
    return stationary_dist


def matrix_exponential(rate_matrix: np.array) -> np.array:
    return torch.matrix_exp(torch.tensor(rate_matrix)).numpy()


def wag_matrix() -> pd.DataFrame():
    wag = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "_rate_matrices/_WAG.txt"), index_col=0, sep="\t")
    pi = compute_stationary_distribution(wag)
    wag_rate = np.dot(-np.diag(wag), pi)
    res = wag / wag_rate
    assert(res.shape == (20, 20))
    return res


def wag_stationary_distribution() -> pd.DataFrame():
    wag = wag_matrix()
    pi = compute_stationary_distribution(wag)
    res = pd.DataFrame(pi, index=wag.index)
    return res


def _composite_index(i, j, num_states):
    return (i - 1) * num_states + j


def chain_product(rate_matrix):
    num_states = rate_matrix.shape[0]
    product_matrix = np.zeros((num_states ** 2, num_states ** 2))
    for i in range(num_states):
        for j in range(num_states):
            for k in range(num_states):
                product_matrix[_composite_index(i, k, num_states), _composite_index(i, j, num_states)] = rate_matrix[k, j]
                product_matrix[_composite_index(k, j, num_states), _composite_index(i, j, num_states)] = rate_matrix[k, i]
    for i in range(num_states):
        for j in range(num_states):
            product_matrix[_composite_index(i, j, num_states), _composite_index(i, j, num_states)] = rate_matrix[i, i] + rate_matrix[j, j]
    return product_matrix
