import os
from typing import Optional

import numpy as np

from src.io import read_count_matrices, read_mask_matrix, write_rate_matrix


def jtt_ipw(
    count_matrices_path: str,
    mask_path: Optional[str],
    use_ipw: bool,
    output_rate_matrix_dir: str,
) -> None:
    # Open frequency matrices
    count_matrices = read_count_matrices(count_matrices_path)
    states = list(count_matrices[0][1].index)
    num_states = len(states)

    if mask_path is not None:
        mask_mat = read_mask_matrix(mask_path).to_numpy()
    else:
        mask_mat = np.ones(shape=(num_states, num_states))

    qtimes, cmats = zip(*count_matrices)
    qtimes = list(qtimes)
    cmats = list(cmats)
    cmats = [cmat.to_numpy() for cmat in cmats]

    # Coalesce transitions a->b and b->a together
    n_time_buckets = cmats[0].shape[0]
    for i in range(n_time_buckets):
        cmats[i] = (cmats[i] + np.transpose(cmats[i])) / 2.0
    # Apply masking
    for i in range(n_time_buckets):
        cmats[i] = cmats[i] * mask_mat

    # Compute CTPs
    # Compute total frequency matrix (ignoring branch lengths)
    F = sum(cmats)
    # Zero the diagonal such that summing over rows will produce the number of
    # transitions from each state.
    F_off = F * (1.0 - np.eye(num_states))
    # Compute CTPs
    CTPs = F_off / (F_off.sum(axis=1)[:, None] + 1e-16)

    # Compute mutabilities
    if use_ipw:
        M = np.zeros(shape=(num_states))
        for i in range(n_time_buckets):
            qtime = qtimes[i]
            cmat = cmats[i]
            cmat_off = cmat * (1.0 - np.eye(num_states))
            M += 1.0 / qtime * cmat_off.sum(axis=1)
        M /= F.sum(axis=1) + 1e-16
    else:
        M = (
            1.0
            / np.median(qtimes)
            * F_off.sum(axis=1)
            / (F.sum(axis=1) + 1e-16)
        )

    # JTT-IPW estimator
    res = np.diag(M) @ CTPs
    np.fill_diagonal(res, -M)

    write_rate_matrix(
        res, states, os.path.join(output_rate_matrix_dir, "result.txt")
    )
