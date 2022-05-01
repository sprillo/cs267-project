import os
import time
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import src.caching as caching
from src.counting import count_transitions
from src.estimation import jtt_ipw, quantized_transitions_mle
from src.io import read_msa, write_contact_map, write_msa
from src.phylogeny_estimation import fast_tree
from src.simulation import simulate_msas
from src.utils import get_amino_acids

MSA_DIR = "/export/home/users/sprillo/Git/Phylo-correction/cs267_data/msas_1024_seqs_None_sites_LG_FastTree.txt-d15ceeb4_RM_20_cats/"
LG_PATH = "./data/rate_matrices/lg.txt"
LG_STATIONARY_PATH = "./data/rate_matrices/lg_stationary.txt"
LG_X_LG_PATH = "./data/rate_matrices/lg_x_lg.txt"
LG_X_LG_STATIONARY_PATH = "./data/rate_matrices/lg_x_lg_stationary.txt"
EQU_PATH = "./data/rate_matrices/equ.txt"

caching.set_cache_dir("_cache")


@caching.cached()
def get_families(
    msa_dir: str,
) -> List[str]:
    """
    Get the list of protein families names.

    Args:
        msa_dir: Directory with the MSA files. There should be one file with
            name family.txt for each protein family.

    Returns:
        The list of protein family names in the provided directory.
    """
    families = sorted(list(os.listdir(msa_dir)))
    families = [x.split(".")[0] for x in families if x.endswith(".txt")]
    return families


@caching.cached()
def get_family_sizes(
    msa_dir: str,
    max_families: Optional[int] = None,
) -> pd.DataFrame:
    """
    Get the size of each protein family.

    By 'size' we mean the number of sequences and the number of sites. These
    are returned in a Pandas DataFrame object, with one row per family.

    Args:
        msa_dir: Directory with the MSA files. There should be one file with
            name family.txt for each protein family.
        max_families: Only return the results for the first `max_families`
            families in lexicographic order.

    Returns:
        A Pandas DataFrame with one row per family, containing the num_sequences
        and num_sites.
    """
    families = get_families(msa_dir=msa_dir)
    if max_families is None:
        max_families = len(families)
    family_size_tuples = []
    for family in families[:max_families]:
        msa = read_msa(os.path.join(msa_dir, f"{family}.txt"))
        num_sequences = len(msa)
        num_sites = len(next(iter(msa.values())))
        family_size_tuples.append((family, num_sequences, num_sites))
    family_size_df = pd.DataFrame(
        family_size_tuples, columns=["family", "num_sequences", "num_sites"]
    )
    return family_size_df


def get_families_within_cutoff(
    msa_dir: str,
    min_num_sites: int,
    max_num_sites: int,
    min_num_sequences: int,
    max_num_sequences: int,
) -> List[str]:
    family_size_df = get_family_sizes(
        msa_dir=msa_dir,
        max_families=None,
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


def end_to_end_simulation(max_families: int, num_processes: int):
    assert max_families <= 1024

    families_all = get_families_within_cutoff(
        msa_dir=MSA_DIR,
        min_num_sites=190,
        max_num_sites=230,
        min_num_sequences=1024,
        max_num_sequences=1024,
    )
    # print(f"len(families_all) = {len(families_all)}")
    # assert False

    families_all = sorted(families_all)
    families_test = families_all[1024:]
    families = families_all[:max_families]

    fast_tree_output_dirs = fast_tree(
        msa_dir=MSA_DIR,
        families=families,
        rate_matrix_path=LG_PATH,
        num_rate_categories=20,
        num_processes=num_processes,
    )

    trivial_contact_map_dir = create_trivial_contact_maps(
        msa_dir=MSA_DIR,
        families=families,
        states=get_amino_acids(),
    )["output_contact_map_dir"]

    simulated_msa_dir = simulate_msas(
        tree_dir=fast_tree_output_dirs["output_tree_dir"],
        site_rates_dir=fast_tree_output_dirs["output_site_rates_dir"],
        contact_map_dir=trivial_contact_map_dir,
        families=families,
        amino_acids=get_amino_acids(),
        pi_1_path=LG_STATIONARY_PATH,
        Q_1_path=LG_PATH,
        pi_2_path=LG_X_LG_STATIONARY_PATH,
        Q_2_path=LG_X_LG_PATH,
        strategy="all_transitions",
        random_seed=0,
        num_processes=num_processes,
        use_cpp_implementation=False,
    )["output_msa_dir"]

    quantization_points = [
        ("%.5f" % (0.06 * 1.1**i)) for i in range(-50, 51, 1)
    ]

    count_matrices__edge_gt__dir = count_transitions(
        tree_dir=fast_tree_output_dirs["output_tree_dir"],
        msa_dir=simulated_msa_dir,
        site_rates_dir=fast_tree_output_dirs["output_site_rates_dir"],
        families=families,
        amino_acids=get_amino_acids(),
        quantization_points=quantization_points,
        edge_or_cherry="edge",
        num_processes=num_processes,
        use_cpp_implementation=False,
    )["output_count_matrices_dir"]

    jtt_ipw__edge_gt__dir = jtt_ipw(
        count_matrices_path=os.path.join(
            count_matrices__edge_gt__dir, "result.txt"
        ),
        mask_path=None,
        use_ipw=True,
        normalize=False,
    )["output_rate_matrix_dir"]

    rate_matrix__edge_gt__dir = quantized_transitions_mle(
        count_matrices_path=os.path.join(
            count_matrices__edge_gt__dir, "result.txt"
        ),
        initialization_path=os.path.join(jtt_ipw__edge_gt__dir, "result.txt"),
        mask_path=None,
        stationary_distribution_path=None,
        rate_matrix_parameterization="pande_reversible",
        device="cpu",
        learning_rate=1e-1,
        num_epochs=200,
        do_adam=True,
    )["output_rate_matrix_dir"]
    print(rate_matrix__edge_gt__dir)

    count_matrices__cherry_gt__dir = count_transitions(
        tree_dir=fast_tree_output_dirs["output_tree_dir"],
        msa_dir=simulated_msa_dir,
        site_rates_dir=fast_tree_output_dirs["output_site_rates_dir"],
        families=families,
        amino_acids=get_amino_acids(),
        quantization_points=quantization_points,
        edge_or_cherry="cherry",
        num_processes=num_processes,
        use_cpp_implementation=False,
    )["output_count_matrices_dir"]

    jtt_ipw__cherry_gt__dir = jtt_ipw(
        count_matrices_path=os.path.join(
            count_matrices__cherry_gt__dir, "result.txt"
        ),
        mask_path=None,
        use_ipw=True,
        normalize=False,
    )["output_rate_matrix_dir"]

    rate_matrix__cherry_gt__dir = quantized_transitions_mle(
        count_matrices_path=os.path.join(
            count_matrices__cherry_gt__dir, "result.txt"
        ),
        initialization_path=os.path.join(jtt_ipw__cherry_gt__dir, "result.txt"),
        mask_path=None,
        stationary_distribution_path=None,
        rate_matrix_parameterization="pande_reversible",
        device="cpu",
        learning_rate=1e-1,
        num_epochs=200,
        do_adam=True,
    )["output_rate_matrix_dir"]
    print(rate_matrix__cherry_gt__dir)

    simulated_msa_leaves_dir = subset_msa_to_leaf_nodes(
        msa_dir=simulated_msa_dir,
        families=families,
        states=get_amino_acids(),
    )["output_msa_dir"]
    print(simulated_msa_leaves_dir)

    fast_tree_output__simulation_iter_1__dirs = fast_tree(
        msa_dir=simulated_msa_leaves_dir,
        families=families,
        rate_matrix_path=EQU_PATH,
        num_rate_categories=20,
        num_processes=num_processes,
    )

    count_matrices__cherry_iter_1__dir = count_transitions(
        tree_dir=fast_tree_output__simulation_iter_1__dirs["output_tree_dir"],
        msa_dir=simulated_msa_leaves_dir,
        site_rates_dir=fast_tree_output__simulation_iter_1__dirs[
            "output_site_rates_dir"
        ],
        families=families,
        amino_acids=get_amino_acids(),
        quantization_points=quantization_points,
        edge_or_cherry="cherry",
        num_processes=num_processes,
        use_cpp_implementation=False,
    )["output_count_matrices_dir"]
    print(count_matrices__cherry_iter_1__dir)

    jtt_ipw__cherry_iter_1__dir = jtt_ipw(
        count_matrices_path=os.path.join(
            count_matrices__cherry_iter_1__dir, "result.txt"
        ),
        mask_path=None,
        use_ipw=True,
        normalize=False,
    )["output_rate_matrix_dir"]
    print(jtt_ipw__cherry_iter_1__dir)

    rate_matrix__cherry_iter_1__dir = quantized_transitions_mle(
        count_matrices_path=os.path.join(
            count_matrices__cherry_iter_1__dir, "result.txt"
        ),
        initialization_path=os.path.join(
            jtt_ipw__cherry_iter_1__dir, "result.txt"
        ),
        mask_path=None,
        stationary_distribution_path=None,
        rate_matrix_parameterization="pande_reversible",
        device="cpu",
        learning_rate=1e-1,
        num_epochs=200,
        do_adam=True,
    )["output_rate_matrix_dir"]
    print(rate_matrix__cherry_iter_1__dir)

    fast_tree_output__simulation_iter_2__dirs = fast_tree(
        msa_dir=simulated_msa_leaves_dir,
        families=families,
        rate_matrix_path=os.path.join(
            rate_matrix__cherry_iter_1__dir, "result.txt"
        ),
        num_rate_categories=20,
        num_processes=num_processes,
    )

    count_matrices__cherry_iter_2__dir = count_transitions(
        tree_dir=fast_tree_output__simulation_iter_2__dirs["output_tree_dir"],
        msa_dir=simulated_msa_leaves_dir,
        site_rates_dir=fast_tree_output__simulation_iter_2__dirs[
            "output_site_rates_dir"
        ],
        families=families,
        amino_acids=get_amino_acids(),
        quantization_points=quantization_points,
        edge_or_cherry="cherry",
        num_processes=num_processes,
        use_cpp_implementation=False,
    )["output_count_matrices_dir"]
    print(count_matrices__cherry_iter_2__dir)

    jtt_ipw__cherry_iter_2__dir = jtt_ipw(
        count_matrices_path=os.path.join(
            count_matrices__cherry_iter_2__dir, "result.txt"
        ),
        mask_path=None,
        use_ipw=True,
        normalize=False,
    )["output_rate_matrix_dir"]
    print(jtt_ipw__cherry_iter_2__dir)

    rate_matrix__cherry_iter_2__dir = quantized_transitions_mle(
        count_matrices_path=os.path.join(
            count_matrices__cherry_iter_2__dir, "result.txt"
        ),
        initialization_path=os.path.join(
            jtt_ipw__cherry_iter_2__dir, "result.txt"
        ),
        mask_path=None,
        stationary_distribution_path=None,
        rate_matrix_parameterization="pande_reversible",
        device="cpu",
        learning_rate=1e-1,
        num_epochs=200,
        do_adam=True,
    )["output_rate_matrix_dir"]
    print(rate_matrix__cherry_iter_2__dir)

    fast_tree_output__simulation_iter_3__dirs = fast_tree(
        msa_dir=simulated_msa_leaves_dir,
        families=families,
        rate_matrix_path=os.path.join(
            rate_matrix__cherry_iter_2__dir, "result.txt"
        ),
        num_rate_categories=20,
        num_processes=num_processes,
    )

    count_matrices__cherry_iter_3__dir = count_transitions(
        tree_dir=fast_tree_output__simulation_iter_3__dirs["output_tree_dir"],
        msa_dir=simulated_msa_leaves_dir,
        site_rates_dir=fast_tree_output__simulation_iter_3__dirs[
            "output_site_rates_dir"
        ],
        families=families,
        amino_acids=get_amino_acids(),
        quantization_points=quantization_points,
        edge_or_cherry="cherry",
        num_processes=num_processes,
        use_cpp_implementation=False,
    )["output_count_matrices_dir"]
    print(count_matrices__cherry_iter_3__dir)

    jtt_ipw__cherry_iter_3__dir = jtt_ipw(
        count_matrices_path=os.path.join(
            count_matrices__cherry_iter_3__dir, "result.txt"
        ),
        mask_path=None,
        use_ipw=True,
        normalize=False,
    )["output_rate_matrix_dir"]
    print(jtt_ipw__cherry_iter_3__dir)

    rate_matrix__cherry_iter_3__dir = quantized_transitions_mle(
        count_matrices_path=os.path.join(
            count_matrices__cherry_iter_3__dir, "result.txt"
        ),
        initialization_path=os.path.join(
            jtt_ipw__cherry_iter_3__dir, "result.txt"
        ),
        mask_path=None,
        stationary_distribution_path=None,
        rate_matrix_parameterization="pande_reversible",
        device="cpu",
        learning_rate=1e-1,
        num_epochs=200,
        do_adam=True,
    )["output_rate_matrix_dir"]
    print(rate_matrix__cherry_iter_3__dir)

    fast_tree_output__simulation_iter_oracle__dirs = fast_tree(
        msa_dir=simulated_msa_leaves_dir,
        families=families,
        rate_matrix_path=LG_PATH,
        num_rate_categories=20,
        num_processes=num_processes,
    )

    count_matrices__cherry_iter_oracle__dir = count_transitions(
        tree_dir=fast_tree_output__simulation_iter_oracle__dirs[
            "output_tree_dir"
        ],
        msa_dir=simulated_msa_leaves_dir,
        site_rates_dir=fast_tree_output__simulation_iter_oracle__dirs[
            "output_site_rates_dir"
        ],
        families=families,
        amino_acids=get_amino_acids(),
        quantization_points=quantization_points,
        edge_or_cherry="cherry",
        num_processes=num_processes,
        use_cpp_implementation=False,
    )["output_count_matrices_dir"]
    print(count_matrices__cherry_iter_oracle__dir)

    jtt_ipw__cherry_iter_oracle__dir = jtt_ipw(
        count_matrices_path=os.path.join(
            count_matrices__cherry_iter_oracle__dir, "result.txt"
        ),
        mask_path=None,
        use_ipw=True,
        normalize=False,
    )["output_rate_matrix_dir"]
    print(jtt_ipw__cherry_iter_oracle__dir)

    rate_matrix__cherry_iter_oracle__dir = quantized_transitions_mle(
        count_matrices_path=os.path.join(
            count_matrices__cherry_iter_oracle__dir, "result.txt"
        ),
        initialization_path=os.path.join(
            jtt_ipw__cherry_iter_oracle__dir, "result.txt"
        ),
        mask_path=None,
        stationary_distribution_path=None,
        rate_matrix_parameterization="pande_reversible",
        device="cpu",
        learning_rate=1e-1,
        num_epochs=200,
        do_adam=True,
    )["output_rate_matrix_dir"]
    print(rate_matrix__cherry_iter_oracle__dir)

    # TODO: Evaluate fit wrt groundh truth using rate matrix metrics, AS WELL AS likelihood on held out families!
    # TODO: rm the fast_tree_log because it has an unnecessarily heavy footprint.
