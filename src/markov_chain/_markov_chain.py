import os

import numpy as np
import pandas as pd
import torch

import src.utils


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
    if torch.cuda.is_available():
        return torch.matrix_exp(torch.tensor(rate_matrix, device='cuda')).cpu().numpy()
    else:
        # TODO: This will use all CPUs by default I think, which is terrible for fast testing on a cluster!
        # Use torch.set_num_threads() before?
        return torch.matrix_exp(torch.tensor(rate_matrix, device='cpu')).numpy()


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


def equ_matrix() -> pd.DataFrame():
    equ = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "_rate_matrices/_Equ.txt"), index_col=0, sep="\t")
    return equ


def equ_stationary_distribution() -> pd.DataFrame():
    pi = [1.0 / 20.0] * 20
    res = pd.DataFrame(pi, index=src.utils.amino_acids)
    return res


def _composite_index(i: int, j: int, num_states: int):
    return i * num_states + j


def chain_product(rate_matrix_1: np.array, rate_matrix_2: np.array) -> np.array:
    assert(rate_matrix_1.shape == rate_matrix_2.shape)
    num_states = rate_matrix_1.shape[0]
    product_matrix = np.zeros((num_states ** 2, num_states ** 2))
    for i in range(num_states):
        for j in range(num_states):
            for k in range(num_states):
                product_matrix[_composite_index(i, k, num_states), _composite_index(i, j, num_states)] = rate_matrix_2[k, j]
                product_matrix[_composite_index(k, j, num_states), _composite_index(i, j, num_states)] = rate_matrix_1[k, i]
    for i in range(num_states):
        for j in range(num_states):
            product_matrix[_composite_index(i, j, num_states), _composite_index(i, j, num_states)] = rate_matrix_1[i, i] + rate_matrix_2[j, j]
    return product_matrix
