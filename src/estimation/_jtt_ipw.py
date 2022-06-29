import logging
import os
import sys
from typing import Optional

import numpy as np

from src import caching
from src.io import read_count_matrices, read_mask_matrix, write_rate_matrix
from src.markov_chain import normalized


def _init_logger():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    fmt_str = "[%(asctime)s] - %(name)s - %(levelname)s - %(message)s"
    formatter = logging.Formatter(fmt_str)

    consoleHandler = logging.StreamHandler(sys.stdout)
    consoleHandler.setFormatter(formatter)
    logger.addHandler(consoleHandler)


_init_logger()


@caching.cached_computation(
    output_dirs=["output_rate_matrix_dir"],
)
def jtt_ipw(
    count_matrices_path: str,
    mask_path: Optional[str],
    use_ipw: bool,
    output_rate_matrix_dir: str,
    normalize: bool = False,
    max_time: Optional[float] = None,
) -> None:
    """
    JTT-IPW estimator.

    Args:
        max_time: Only data from transitions with length <= max_time will be
            used to compute the estimator. The estimator works best on short
            transitions, which poses a bias-variance tradeoff.
    """
    logger = logging.getLogger(__name__)
    logger.info("Starting")

    # Open frequency matrices
    count_matrices = read_count_matrices(count_matrices_path)
    states = list(count_matrices[0][1].index)
    num_states = len(states)

    if mask_path is not None:
        mask_mat = read_mask_matrix(mask_path).to_numpy()
    else:
        mask_mat = np.ones(shape=(num_states, num_states))

    qtimes, cmats = zip(*count_matrices)
    del count_matrices
    qtimes = list(qtimes)
    cmats = list(cmats)
    if max_time is not None:
        valid_time_indices = [
            i for i in range(len(qtimes)) if qtimes[i] <= max_time
        ]
        qtimes = [qtimes[i] for i in valid_time_indices]
        cmats = [cmats[i] for i in valid_time_indices]
    cmats = [cmat.to_numpy() for cmat in cmats]

    # Coalesce transitions a->b and b->a together
    n_time_buckets = len(cmats)
    assert cmats[0].shape == (num_states, num_states)
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

    if normalize:
        res = normalized(res)

    write_rate_matrix(
        res, states, os.path.join(output_rate_matrix_dir, "result.txt")
    )

    logger.info("Done!")
