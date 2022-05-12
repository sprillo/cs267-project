import numpy as np

RateMatrixType = np.array


def _masked_log_ratio(
    y: RateMatrixType,
    y_hat: RateMatrixType,
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
    return masked_log_ratio


def l_infty_norm(
    y: RateMatrixType,
    y_hat: RateMatrixType,
) -> float:
    masked_log_ratio = _masked_log_ratio(y, y_hat)
    res = np.max(np.abs(masked_log_ratio))
    return res


def rmse(
    y: RateMatrixType,
    y_hat: RateMatrixType,
) -> float:
    num_states = y.shape[0]
    masked_log_ratio = _masked_log_ratio(y, y_hat)
    masked_log_ratio_squared = masked_log_ratio * masked_log_ratio
    res = np.sqrt(
        np.sum(masked_log_ratio_squared) / (num_states * (num_states - 1.0))
    )
    return res


def mre(
    y: RateMatrixType,
    y_hat: RateMatrixType,
) -> float:
    """
    Max relative error.
    """
    return np.exp(l_infty_norm(y, y_hat)) - 1
