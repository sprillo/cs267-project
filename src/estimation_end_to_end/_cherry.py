import os
from typing import Dict, List, Optional, Tuple

from src.counting import count_co_transitions, count_transitions
from src.estimation import jtt_ipw, quantized_transitions_mle
from src.types import PhylogenyEstimatorType
from src.utils import get_amino_acids
from src.markov_chain import get_equ_path, get_equ_x_equ_path
from src.evaluation import create_maximal_matching_contact_map
from src.benchmarking.pfam_15k import compute_contact_maps


def cherry_estimator(
    msa_dir: str,
    families: List[str],
    tree_estimator: PhylogenyEstimatorType,
    initial_tree_estimator_rate_matrix_path: str,
    num_iterations: int,
    num_processes: Optional[int] = 2,
    quantization_grid_center: float = 0.03,
    quantization_grid_step: float = 1.1,
    quantization_grid_num_steps: int = 64,
    use_cpp_counting_implementation: bool = True,
    device: str = "cpu",
    learning_rate: float = 1e-1,
    num_epochs: int = 2000,
    do_adam: bool = True,
    edge_or_cherry: str = "cherry",
    cpp_counting_command_line_prefix: str = "",
    cpp_counting_command_line_suffix: str = "",
    num_processes_tree_estimation: Optional[int] = None,
    num_processes_counting: Optional[int] = None,
    num_processes_optimization: Optional[int] = 2,
    optimizer_initialization: str = "jtt-ipw",
    optimizer_return_best_iter: bool = True,
) -> Dict:
    """
    Cherry estimator.

    Returns a dictionary with the directories to the intermediate outputs. In
    particular, the learned rate matrix is indexed by "learned_rate_matrix_path"
    """
    if num_processes_tree_estimation is None:
        num_processes_tree_estimation = num_processes
    if num_processes_counting is None:
        num_processes_counting = num_processes
    if num_processes_optimization is None:
        num_processes_optimization = num_processes

    res = {}

    quantization_points = [
        ("%.5f" % (quantization_grid_center * quantization_grid_step**i))
        for i in range(
            -quantization_grid_num_steps, quantization_grid_num_steps + 1, 1
        )
    ]

    res["quantization_points"] = quantization_points

    current_estimate_rate_matrix_path = initial_tree_estimator_rate_matrix_path
    for iteration in range(num_iterations):
        tree_estimator_output_dirs = tree_estimator(
            msa_dir=msa_dir,
            families=families,
            rate_matrix_path=current_estimate_rate_matrix_path,
            num_processes=num_processes_tree_estimation,
        )
        res[
            f"tree_estimator_output_dirs_{iteration}"
        ] = tree_estimator_output_dirs

        count_matrices_dir = count_transitions(
            tree_dir=tree_estimator_output_dirs["output_tree_dir"],
            msa_dir=msa_dir,
            site_rates_dir=tree_estimator_output_dirs["output_site_rates_dir"],
            families=families,
            amino_acids=get_amino_acids(),
            quantization_points=quantization_points,
            edge_or_cherry=edge_or_cherry,
            num_processes=num_processes_counting,
            use_cpp_implementation=use_cpp_counting_implementation,
            cpp_command_line_prefix=cpp_counting_command_line_prefix,
            cpp_command_line_suffix=cpp_counting_command_line_suffix,
        )["output_count_matrices_dir"]

        res[f"count_matrices_dir_{iteration}"] = count_matrices_dir

        jtt_ipw_dir = jtt_ipw(
            count_matrices_path=os.path.join(count_matrices_dir, "result.txt"),
            mask_path=None,
            use_ipw=True,
            normalize=False,
        )["output_rate_matrix_dir"]

        res[f"jtt_ipw_dir_{iteration}"] = jtt_ipw_dir

        initialization_path = None
        if optimizer_initialization == "jtt-ipw":
            initialization_path = os.path.join(jtt_ipw_dir, "result.txt")
        elif optimizer_initialization == "equ":
            initialization_path = get_equ_path()
        elif optimizer_initialization == "random":
            initialization_path = None
        else:
            raise ValueError(f"Uknown optimizer_initialization = {optimizer_initialization}")

        rate_matrix_dir = quantized_transitions_mle(
            count_matrices_path=os.path.join(count_matrices_dir, "result.txt"),
            initialization_path=initialization_path,
            mask_path=None,
            stationary_distribution_path=None,
            rate_matrix_parameterization="pande_reversible",
            device=device,
            learning_rate=learning_rate,
            num_epochs=num_epochs,
            do_adam=do_adam,
            OMP_NUM_THREADS=num_processes_optimization,
            OPENBLAS_NUM_THREADS=num_processes_optimization,
            return_best_iter=optimizer_return_best_iter,
        )["output_rate_matrix_dir"]

        res[f"rate_matrix_dir_{iteration}"] = rate_matrix_dir

        current_estimate_rate_matrix_path = os.path.join(
            rate_matrix_dir, "result.txt"
        )

    res["learned_rate_matrix_path"] = current_estimate_rate_matrix_path

    return res


def cherry_estimator_coevolution(
    msa_dir: str,
    contact_map_dir: str,
    minimum_distance_for_nontrivial_contact: int,
    coevolution_mask_path: str,
    families: List[str],
    tree_estimator: PhylogenyEstimatorType,
    initial_tree_estimator_rate_matrix_path: str,
    # num_iterations: int,  # There is no iteration in this case!
    num_processes: int = 8,
    quantization_grid_center: float = 0.03,
    quantization_grid_step: float = 1.1,
    quantization_grid_num_steps: int = 64,
    use_cpp_counting_implementation: bool = True,
    device: str = "cpu",
    learning_rate: float = 1e-1,
    num_epochs: int = 500,
    do_adam: bool = True,
    edge_or_cherry: str = "cherry",
    cpp_counting_command_line_prefix: str = "",
    cpp_counting_command_line_suffix: str = "",
    num_processes_tree_estimation: Optional[int] = None,
    num_processes_counting: Optional[int] = None,
    num_processes_optimization: Optional[int] = 8,
    optimizer_initialization: str = "jtt-ipw",
    optimizer_return_best_iter: bool = True,
    use_maximal_matching: bool = False,
) -> Dict:
    """
    Cherry estimator for coevolution.

    Returns a dictionary with the directories to the intermediate outputs. In
    particular, the learned coevolution rate matrix is indexed by
    "learned_rate_matrix_path"
    """
    if num_processes_tree_estimation is None:
        num_processes_tree_estimation = num_processes
    if num_processes_counting is None:
        num_processes_counting = num_processes
    if num_processes_optimization is None:
        num_processes_optimization = num_processes

    res = {}

    quantization_points = [
        ("%.5f" % (quantization_grid_center * quantization_grid_step**i))
        for i in range(
            -quantization_grid_num_steps, quantization_grid_num_steps + 1, 1
        )
    ]

    res["quantization_points"] = quantization_points

    current_estimate_rate_matrix_path = initial_tree_estimator_rate_matrix_path
    for iteration in range(1):  # There is no iteration in this case.
        tree_estimator_output_dirs = tree_estimator(
            msa_dir=msa_dir,
            families=families,
            rate_matrix_path=current_estimate_rate_matrix_path,
            num_processes=num_processes_tree_estimation,
        )

        res[
            f"tree_estimator_output_dirs_{iteration}"
        ] = tree_estimator_output_dirs

        mdnc = minimum_distance_for_nontrivial_contact

        if use_maximal_matching:
            # Need to compute a maximal matching instead of using the whole
            # contact maps
            contact_map_dir = create_maximal_matching_contact_map(
                i_contact_map_dir=contact_map_dir,
                families=families,
                minimum_distance_for_nontrivial_contact=mdnc,
                num_processes=num_processes_counting,
            )["o_contact_map_dir"]

        count_matrices_dir = count_co_transitions(
            tree_dir=tree_estimator_output_dirs["output_tree_dir"],
            msa_dir=msa_dir,
            contact_map_dir=contact_map_dir,
            families=families,
            amino_acids=get_amino_acids(),
            quantization_points=quantization_points,
            edge_or_cherry=edge_or_cherry,
            minimum_distance_for_nontrivial_contact=mdnc,
            num_processes=num_processes_counting,
            use_cpp_implementation=use_cpp_counting_implementation,
            cpp_command_line_prefix=cpp_counting_command_line_prefix,
            cpp_command_line_suffix=cpp_counting_command_line_suffix,
        )["output_count_matrices_dir"]

        res[f"count_matrices_dir_{iteration}"] = count_matrices_dir

        jtt_ipw_dir = jtt_ipw(
            count_matrices_path=os.path.join(count_matrices_dir, "result.txt"),
            mask_path=coevolution_mask_path,
            use_ipw=True,
            normalize=False,
        )["output_rate_matrix_dir"]

        res[f"jtt_ipw_dir_{iteration}"] = jtt_ipw_dir

        initialization_path = None
        if optimizer_initialization == "jtt-ipw":
            initialization_path = os.path.join(jtt_ipw_dir, "result.txt")
        elif optimizer_initialization == "equ_x_equ":
            initialization_path = get_equ_x_equ_path()
        elif optimizer_initialization == "random":
            initialization_path = None
        else:
            raise ValueError(f"Uknown optimizer_initialization = {optimizer_initialization}")

        rate_matrix_dir = quantized_transitions_mle(
            count_matrices_path=os.path.join(count_matrices_dir, "result.txt"),
            initialization_path=initialization_path,
            mask_path=coevolution_mask_path,
            stationary_distribution_path=None,
            rate_matrix_parameterization="pande_reversible",
            device=device,
            learning_rate=learning_rate,
            num_epochs=num_epochs,
            do_adam=do_adam,
            OMP_NUM_THREADS=num_processes_optimization,
            OPENBLAS_NUM_THREADS=num_processes_optimization,
            return_best_iter=optimizer_return_best_iter,
        )["output_rate_matrix_dir"]

        res[f"rate_matrix_dir_{iteration}"] = rate_matrix_dir

        current_estimate_rate_matrix_path = os.path.join(
            rate_matrix_dir, "result.txt"
        )

    res["learned_rate_matrix_path"] = current_estimate_rate_matrix_path

    return res
