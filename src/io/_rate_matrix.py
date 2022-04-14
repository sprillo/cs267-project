import pandas as pd


def write_probability_distribution(
    probability_distribution: pd.DataFrame,
    probability_distribution_path: str,
) -> None:
    rate_matrix.to_csv(
        probability_distribution,
        probability_distribution_path,
        sep="\t",
        index=True,
    )


def write_rate_matrix(
    rate_matrix: pd.DataFrame,
    rate_matrix_path: str,
) -> None:
    rate_matrix.to_csv(
        rate_matrix,
        rate_matrix_path,
        sep="\t",
        index=True,
    )


def read_rate_matrix(rate_matrix_path: str) -> pd.DataFrame:
    res = pd.read_csv(
        rate_matrix_path,
        delim_whitespace=True,
        index_col=0,
        # dtype=float,
    ).astype(float)
    # TODO: Assert that it is a rate matrix
    return res


def read_probability_distribution(
    probability_distribution_path: str,
) -> pd.DataFrame:
    res = pd.read_csv(
        probability_distribution_path,
        delim_whitespace=True,
        index_col=0,
    ).astype(float)
    if res.shape[1] != 1:
        raise Exception(
            f"Probability distribution at {probability_distribution_path} "
            "should be one-dimensional."
        )
    if abs(res.sum().sum() - 1.0) > 1e-6:
        raise Exception(
            f"Probability distribution at {probability_distribution_path} "
            "should add to 1.0, with a tolerance of 1e-6."
        )
    return res
