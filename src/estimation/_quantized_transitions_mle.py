import tempfile
from typing import Optional

from ._ratelearn import RateMatrixLearner


def quantized_transitions_mle(
    count_matrices_path: str,
    initialization_path: Optional[str],
    mask_path: Optional[str],
    output_rate_matrix_dir: str,
    stationary_distribution_path: Optional[str] = None,
    rate_matrix_parameterization: str = "pande_reversible",
    device: str = "cpu",
    learning_rate: float = 1e-1,
    num_epochs: int = 2000,
    do_adam: bool = True,
):
    assert device in ["cpu", "cuda"]
    raise NotImplementedError
    with tempfile.NamedTemporaryFile("w") as frequency_matrices_file:
        frequency_matrices_path = frequency_matrices_file.name
        with tempfile.NamedTemporaryFile("w") as mask2_file:
            mask2_path = mask2_file.name
            with tempfile.TemporaryDirectory() as output_dir:
                # TODO: Fill in frequency_matrices_path
                # TODO: Convert mask to mask2
                # initialization = TODO
                rate_matrix_learner = RateMatrixLearner(
                    frequency_matrices=frequency_matrices_path,  # "test_input_data/matrices_toy.txt",
                    output_dir=output_dir,
                    stationnary_distribution=None,
                    mask=mask2_path,  # "test_input_data/3x3_mask.txt",
                    rate_matrix_parameterization=rate_matrix_parameterization,
                    device=device,
                    initialization=initialization,  # np.loadtxt("test_input_data/3x3_pande_reversible_initialization.txt"),
                )
                rate_matrix_learner.train(
                    lr=learning_rate,
                    num_epochs=num_epochs,
                    do_adam=do_adam,
                )
