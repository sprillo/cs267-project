import os
from typing import List

from src.counting import count_transitions
from src.estimation import jtt_ipw, quantized_transitions_mle
from src.phylogeny_estimation import fast_tree
from src.utils import get_amino_acids


def cherry_estimator(
    msa_dir: str,
    families: List[str],
    initial_rate_matrix_path: str,
    num_rate_categories: int,
    num_iterations: int,
    num_processes: int,
    quantization_grid_center: float = 0.06,
    quantization_grid_step: float = 1.1,
    quantization_grid_num_steps: int = 50,
) -> str:
    """
    Cherry estimator.

    Returns the path to the estimated rate matrix.
    """
    quantization_points = [
        ("%.5f" % (quantization_grid_center * quantization_grid_step**i))
        for i in range(
            -quantization_grid_num_steps, quantization_grid_num_steps + 1, 1
        )
    ]

    current_estimate_rate_matrix_path = initial_rate_matrix_path
    for iteration in range(num_iterations):
        fast_tree_output_dirs = fast_tree(
            msa_dir=msa_dir,
            families=families,
            rate_matrix_path=current_estimate_rate_matrix_path,
            num_rate_categories=num_rate_categories,
            num_processes=num_processes,
        )

        count_matrices_dir = count_transitions(
            tree_dir=fast_tree_output_dirs["output_tree_dir"],
            msa_dir=msa_dir,
            site_rates_dir=fast_tree_output_dirs["output_site_rates_dir"],
            families=families,
            amino_acids=get_amino_acids(),
            quantization_points=quantization_points,
            edge_or_cherry="cherry",
            num_processes=num_processes,
            use_cpp_implementation=False,
        )["output_count_matrices_dir"]

        jtt_ipw_dir = jtt_ipw(
            count_matrices_path=os.path.join(count_matrices_dir, "result.txt"),
            mask_path=None,
            use_ipw=True,
            normalize=False,
        )["output_rate_matrix_dir"]

        rate_matrix_dir = quantized_transitions_mle(
            count_matrices_path=os.path.join(count_matrices_dir, "result.txt"),
            initialization_path=os.path.join(jtt_ipw_dir, "result.txt"),
            mask_path=None,
            stationary_distribution_path=None,
            rate_matrix_parameterization="pande_reversible",
            device="cpu",
            learning_rate=1e-1,
            num_epochs=200,
            do_adam=True,
        )["output_rate_matrix_dir"]

        current_estimate_rate_matrix_path = os.path.join(
            rate_matrix_dir, "result.txt"
        )
    return current_estimate_rate_matrix_path
