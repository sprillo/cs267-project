from typing import Optional

import numpy as np

RateMatrixType = np.array
MaskMatrixType = np.array


def _masked_log_ratio(
    y: RateMatrixType,
    y_hat: RateMatrixType,
    mask_matrix: Optional[MaskMatrixType] = None,
) -> np.array:
    if y.shape != y_hat.shape:
        raise ValueError(
            "y and y_hat should have the same shape. Shapes are: "
            f"y.shape={y.shape}, y_hat.shape={y_hat.shape}"
        )
    num_states = y.shape[0]
    assert y.shape == (num_states, num_states)
    assert y_hat.shape == (num_states, num_states)
    off_diag_mask = 1 - np.eye(num_states, num_states)
    ratio = y / y_hat
    log_ratio = np.log(ratio)
    masked_log_ratio = log_ratio * off_diag_mask
    if mask_matrix is not None:
        for i in range(num_states):
            for j in range(num_states):
                if mask_matrix[i, j] == 0:
                    masked_log_ratio[i, j] = 0
    return masked_log_ratio


def l_infty_norm(
    y: RateMatrixType,
    y_hat: RateMatrixType,
    mask_matrix: Optional[MaskMatrixType] = None,
) -> float:
    masked_log_ratio = _masked_log_ratio(y, y_hat, mask_matrix)
    res = np.max(np.abs(masked_log_ratio))
    return res


def rmse(
    y: RateMatrixType,
    y_hat: RateMatrixType,
    mask_matrix: Optional[MaskMatrixType] = None,
) -> float:
    num_states = y.shape[0]
    masked_log_ratio = _masked_log_ratio(y, y_hat, mask_matrix)
    masked_log_ratio_squared = masked_log_ratio * masked_log_ratio
    if mask_matrix is not None:
        total_non_masked_entries = (
            mask_matrix.sum().sum() - num_states
        )  # Need to remove the diagonal
    else:
        total_non_masked_entries = num_states * (num_states - 1)
    res = np.sqrt(np.sum(masked_log_ratio_squared) / total_non_masked_entries)
    return res


def mre(
    y: RateMatrixType,
    y_hat: RateMatrixType,
    mask_matrix: Optional[MaskMatrixType] = None,
) -> float:
    """
    Max relative error.
    """
    return np.exp(l_infty_norm(y, y_hat, mask_matrix)) - 1
