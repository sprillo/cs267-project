import logging
import os
import sys
import time
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import caching
from src.benchmarking.pfam_15k import subsample_pfam_15k_msas
from src.caching import secure_parallel_output
from src.io import read_msa, write_contact_map, write_msa
from src.markov_chain import (
    get_lg_path,
    get_lg_stationary_path,
    get_lg_x_lg_path,
    get_lg_x_lg_stationary_path,
)
from src.phylogeny_estimation import fast_tree
from src.simulation import simulate_msas
from src.utils import get_amino_acids


def _init_logger():
    logger = logging.getLogger("benchmarking.simulated_data_benchmarking")
    logger.setLevel(logging.INFO)
    fmt_str = "[%(asctime)s] - %(name)s - %(levelname)s - %(message)s"
    formatter = logging.Formatter(fmt_str)

    consoleHandler = logging.StreamHandler(sys.stdout)
    consoleHandler.setFormatter(formatter)
    logger.addHandler(consoleHandler)


_init_logger()


@caching.cached()
def get_families(
    pfam_15k_msa_dir: str,
) -> List[str]:
    """
    Get the list of protein families names.

    Args:
        pfam_15k_msa_dir: Directory with the MSA files. There should be one
            file with name family.txt for each protein family.

    Returns:
        The list of protein family names in the provided directory.
    """
    families = sorted(list(os.listdir(pfam_15k_msa_dir)))
    families = [x.split(".")[0] for x in families if x.endswith(".a3m")]
    return families


@caching.cached()
def get_family_sizes(
    pfam_15k_msa_dir: str,
) -> pd.DataFrame:
    """
    Get the size of each protein family.

    By 'size' we mean the number of sequences and the number of sites. These
    are returned in a Pandas DataFrame object, with one row per family.

    Args:
        pfam_15k_msa_dir: Directory with the MSA files. There should be one
            file with name family.a3m for each protein family.

    Returns:
        A Pandas DataFrame with one row per family, containing the num_sequences
        and num_sites.
    """
    families = get_families(pfam_15k_msa_dir=pfam_15k_msa_dir)
    family_size_tuples = []
    for family in families:
        for i, line in enumerate(
            open(os.path.join(pfam_15k_msa_dir, f"{family}.a3m"), "r")
        ):
            if i == 1:
                num_sites = len(line.strip())
        num_lines = (
            open(os.path.join(pfam_15k_msa_dir, f"{family}.a3m"), "r")
            .read()
            .strip()
            .count("\n")
            + 1
        )
        assert num_lines % 2 == 0
        num_sequences = num_lines // 2
        family_size_tuples.append((family, num_sequences, num_sites))
    family_size_df = pd.DataFrame(
        family_size_tuples, columns=["family", "num_sequences", "num_sites"]
    )
    return family_size_df


def get_families_within_cutoff(
    pfam_15k_msa_dir: str,
    min_num_sites: int,
    max_num_sites: int,
    min_num_sequences: int,
    max_num_sequences: int,
) -> List[str]:
    family_size_df = get_family_sizes(
        pfam_15k_msa_dir=pfam_15k_msa_dir,
    )
    families = family_size_df.family[
        (family_size_df.num_sites >= min_num_sites)
        & (family_size_df.num_sites <= max_num_sites)
        & (family_size_df.num_sequences >= min_num_sequences)
        & (family_size_df.num_sequences <= max_num_sequences)
    ]
    families = list(families)
    return families


def fig_family_sizes(
    msa_dir: str,
    max_families: Optional[int] = None,
) -> None:
    """
    Histograms of num_sequences and num_sites.
    """
    family_size_df = get_family_sizes(
        msa_dir=msa_dir,
        max_families=max_families,
    )
    plt.title("Distribution of family num_sequences")
    plt.hist(family_size_df.num_sequences)
    plt.show()

    plt.title("Distribution of family num_sites")
    print(f"median num_sites = {family_size_df.num_sites.median()}")
    print(f"mode num_sites = {family_size_df.num_sites.mode()}")
    print(f"mean num_sites = {family_size_df.num_sites.mean()}")
    plt.hist(family_size_df.num_sites, bins=100)
    plt.show()

    plt.xlabel("num_sequences")
    plt.ylabel("num_sites")
    plt.scatter(
        family_size_df.num_sequences, family_size_df.num_sites, alpha=0.3
    )
    plt.show()


@caching.cached_parallel_computation(
    parallel_arg="families",
    output_dirs=["output_contact_map_dir"],
)
def create_trivial_contact_maps(
    msa_dir: str,
    families: List[str],
    states: List[str],
    output_contact_map_dir: Optional[str] = None,
):
    for family in families:
        st = time.time()
        msa = read_msa(os.path.join(msa_dir, family + ".txt"))
        num_sites = len(next(iter(msa.values())))
        contact_map = np.zeros(shape=(num_sites, num_sites), dtype=int)
        write_contact_map(
            contact_map, os.path.join(output_contact_map_dir, family + ".txt")
        )
        et = time.time()
        open(
            os.path.join(output_contact_map_dir, family + ".profiling"), "w"
        ).write(f"Total time: {et - st}\n")
        secure_parallel_output(output_contact_map_dir, family)


@caching.cached_parallel_computation(
    parallel_arg="families",
    output_dirs=["output_msa_dir"],
)
def subset_msa_to_leaf_nodes(
    msa_dir: str,
    families: List[str],
    states: List[str],
    output_msa_dir: Optional[str] = None,
):
    """
    An internal node is anyone that starts with 'internal-'.
    """
    for family in families:
        msa = read_msa(os.path.join(msa_dir, family + ".txt"))
        msa_subset = {
            seq_name: seq
            for (seq_name, seq) in msa.items()
            if not seq_name.startswith("internal-")
        }
        write_msa(msa_subset, os.path.join(output_msa_dir, family + ".txt"))
        secure_parallel_output(output_msa_dir, family)


def simulate_ground_truth_data_single_site(
    pfam_15k_msa_dir: str,
    families: List[str],
    num_sequences: int,
    num_rate_categories: int,
    num_processes: int,
    random_seed: int,
):
    """
    Simulate ground truth MSAs with LG.
    """
    real_msa_dir = subsample_pfam_15k_msas(
        pfam_15k_msa_dir=pfam_15k_msa_dir,
        num_sequences=num_sequences,
        families=families,
        num_processes=num_processes,
    )["output_msa_dir"]

    # contact_map_dir = compute_contact_maps(
    #     pfam_15k_pdb_dir=PFAM_15K_PDB_DIR,
    #     families=families,
    #     angstrom_cutoff=8.0,
    #     num_processes=num_processes,
    # )["output_contact_map_dir"]

    fast_tree_output = fast_tree(
        msa_dir=real_msa_dir,
        families=families,
        rate_matrix_path=get_lg_path(),
        num_rate_categories=num_rate_categories,
        num_processes=num_processes,
    )

    gt_trees, gt_site_rates, gt_likelihood_dir = (
        fast_tree_output["output_tree_dir"],
        fast_tree_output["output_site_rates_dir"],
        fast_tree_output["output_likelihood_dir"],
    )

    # We only investigate single-site model here.
    contact_map_dir = create_trivial_contact_maps(
        msa_dir=real_msa_dir,
        families=families,
        states=get_amino_acids(),
    )["output_contact_map_dir"]

    # Now we simulate MSAs
    gt_msa_dir = simulate_msas(
        tree_dir=gt_trees,
        site_rates_dir=gt_site_rates,
        contact_map_dir=contact_map_dir,
        families=families,
        amino_acids=get_amino_acids(),
        pi_1_path=get_lg_stationary_path(),
        Q_1_path=get_lg_path(),
        pi_2_path=get_lg_x_lg_stationary_path(),
        Q_2_path=get_lg_x_lg_path(),
        strategy="all_transitions",
        random_seed=random_seed,
        num_processes=num_processes,
        use_cpp_implementation=False,
    )["output_msa_dir"]

    # Now subset the MSAs to only the leaf nodes.
    msa_dir = subset_msa_to_leaf_nodes(
        msa_dir=gt_msa_dir,
        families=families,
        states=get_amino_acids(),
    )["output_msa_dir"]

    return (
        msa_dir,
        contact_map_dir,
        gt_msa_dir,
        gt_trees,
        gt_site_rates,
        gt_likelihood_dir,
    )


# def end_to_end_simulation_single_site(
#     msa_dir: str,
#     num_training_families: int,
#     num_processes: int,
#     num_sequences: int = 128,
#     num_rate_categories: int = 20,
# ):

#     tree_dir, site_rates_dir, contact_map_dir, msa_dir, msa_leaves_dir = simulate_ground_truth_data(
#         msa_dir=msa_dir,
#         num_processes=num_processes,
#         num_sequences=num_sequences,
#         num_rate_categories=num_rate_categories,
#     )

#     quantization_points = [
#         ("%.5f" % (0.06 * 1.1**i)) for i in range(-50, 51, 1)
#     ]

#     # Run the oracle method
#     count_matrices__edge_gt__dir = count_transitions(
#         tree_dir=fast_tree_output_dirs["output_tree_dir"],
#         msa_dir=simulated_msa_dir,
#         site_rates_dir=fast_tree_output_dirs["output_site_rates_dir"],
#         families=families_train,
#         amino_acids=get_amino_acids(),
#         quantization_points=quantization_points,
#         edge_or_cherry="edge",
#         num_processes=num_processes,
#         use_cpp_implementation=False,
#     )["output_count_matrices_dir"]

#     jtt_ipw__edge_gt__dir = jtt_ipw(
#         count_matrices_path=os.path.join(
#             count_matrices__edge_gt__dir, "result.txt"
#         ),
#         mask_path=None,
#         use_ipw=True,
#         normalize=False,
#     )["output_rate_matrix_dir"]

#     rate_matrix__edge_gt__dir = quantized_transitions_mle(
#         count_matrices_path=os.path.join(
#             count_matrices__edge_gt__dir, "result.txt"
#         ),
#         initialization_path=os.path.join(jtt_ipw__edge_gt__dir, "result.txt"),
#         mask_path=None,
#         stationary_distribution_path=None,
#         rate_matrix_parameterization="pande_reversible",
#         device="cpu",
#         learning_rate=1e-1,
#         num_epochs=200,
#         do_adam=True,
#     )["output_rate_matrix_dir"]
#     print(rate_matrix__edge_gt__dir)

#     # Run Cherry on GT trees
#     count_matrices__cherry_gt__dir = count_transitions(
#         tree_dir=fast_tree_output_dirs["output_tree_dir"],
#         msa_dir=simulated_msa_dir,
#         site_rates_dir=fast_tree_output_dirs["output_site_rates_dir"],
#         families=families_train,
#         amino_acids=get_amino_acids(),
#         quantization_points=quantization_points,
#         edge_or_cherry="cherry",
#         num_processes=num_processes,
#         use_cpp_implementation=False,
#     )["output_count_matrices_dir"]

#     jtt_ipw__cherry_gt__dir = jtt_ipw(
#         count_matrices_path=os.path.join(
#             count_matrices__cherry_gt__dir, "result.txt"
#         ),
#         mask_path=None,
#         use_ipw=True,
#         normalize=False,
#     )["output_rate_matrix_dir"]

#     rate_matrix__cherry_gt__dir = quantized_transitions_mle(
#         count_matrices_path=os.path.join(
#             count_matrices__cherry_gt__dir, "result.txt"
#         ),
#         initialization_path=os.path.join(jtt_ipw__cherry_gt__dir, "result.txt"),
#         mask_path=None,
#         stationary_distribution_path=None,
#         rate_matrix_parameterization="pande_reversible",
#         device="cpu",
#         learning_rate=1e-1,
#         num_epochs=200,
#         do_adam=True,
#     )["output_rate_matrix_dir"]

#     # Do Cherry method!
#     cherry_rate_matrix_dir = cherry_estimator(
#         msa_dir=simulated_msa_leaves_dir,
#         families=families_train,
#         initial_rate_matrix_path=EQU_PATH,
#         num_rate_categories=4,
#     )["output_rate_matrix_dir"]

#     # Do cherry but with oracle rate matrix in FastTree, to see how close 3
#     # iterations is to oracle rate matrix in LG (shouldn't get better than that)
#     fast_tree_output__simulation_iter_oracle__dirs = fast_tree(
#         msa_dir=simulated_msa_leaves_dir,
#         families=families,
#         rate_matrix_path=LG_PATH,
#         num_rate_categories=20,
#         num_processes=num_processes,
#     )

#     count_matrices__cherry_iter_oracle__dir = count_transitions(
#         tree_dir=fast_tree_output__simulation_iter_oracle__dirs[
#             "output_tree_dir"
#         ],
#         msa_dir=simulated_msa_leaves_dir,
#         site_rates_dir=fast_tree_output__simulation_iter_oracle__dirs[
#             "output_site_rates_dir"
#         ],
#         families=families,
#         amino_acids=get_amino_acids(),
#         quantization_points=quantization_points,
#         edge_or_cherry="cherry",
#         num_processes=num_processes,
#         use_cpp_implementation=False,
#     )["output_count_matrices_dir"]

#     jtt_ipw__cherry_iter_oracle__dir = jtt_ipw(
#         count_matrices_path=os.path.join(
#             count_matrices__cherry_iter_oracle__dir, "result.txt"
#         ),
#         mask_path=None,
#         use_ipw=True,
#         normalize=False,
#     )["output_rate_matrix_dir"]

#     rate_matrix__cherry_iter_oracle__dir = quantized_transitions_mle(
#         count_matrices_path=os.path.join(
#             count_matrices__cherry_iter_oracle__dir, "result.txt"
#         ),
#         initialization_path=os.path.join(
#             jtt_ipw__cherry_iter_oracle__dir, "result.txt"
#         ),
#         mask_path=None,
#         stationary_distribution_path=None,
#         rate_matrix_parameterization="pande_reversible",
#         device="cpu",
#         learning_rate=1e-1,
#         num_epochs=200,
#         do_adam=True,
#     )["output_rate_matrix_dir"]

#     # TODO: Evaluate fit wrt groundh truth using rate matrix metrics, AS WELL AS likelihood on held out families!
#     # TODO: rm the fast_tree_log because it has an unnecessarily heavy footprint.
