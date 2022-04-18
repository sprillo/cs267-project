import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from src.io import (
    Tree,
    read_contact_map,
    read_msa,
    read_probability_distribution,
    read_rate_matrix,
    read_site_rates,
    read_tree,
    write_log_likelihood,
)
from src.markov_chain import FactorizedReversibleModel, matrix_exponential


def dp_likelihood_computation(
    tree: Tree,
    msa: Dict[str, str],
    contact_map: np.array,
    site_rates: List[float],
    amino_acids: List[str],
    pi_1: np.array,
    Q_1: np.array,
    fact_1: Optional[FactorizedReversibleModel],
    reversible_1: bool,
    device_1: bool,
    pi_2: np.array,
    Q_2: np.array,
    fact_2: Optional[FactorizedReversibleModel],
    reversible_2: bool,
    device_2: bool,
) -> Tuple[float, List[float]]:
    """
    Compute the data log-likelihood with dynamic programming.
    """
    st_all = time.time()
    st = time.time()
    # These are just binary vectors that encode the observation sets
    node_observations_single_site = {}  # type: Dict[str, np.array]
    node_observations_pair_site = {}  # type: Dict[str, np.array]

    pairs_of_amino_acids = [
        aa1 + aa2 for aa1 in amino_acids for aa2 in amino_acids
    ]

    num_sites = len(site_rates)

    contacting_pairs = list(zip(*np.where(contact_map == 1)))
    contacting_pairs = [(i, j) for (i, j) in contacting_pairs if i < j]
    # Validate that each site is in contact with at most one other site
    contacting_sites = list(sum(contacting_pairs, ()))
    if len(set(contacting_sites)) != len(contacting_sites):
        raise Exception(
            f"Each site can only be in contact with one other site. "
            f"The contacting sites were: {contacting_pairs}"
        )
    independent_sites = [
        i for i in range(num_sites) if i not in contacting_sites
    ]

    n_independent_sites = len(independent_sites)
    n_contacting_pairs = len(contacting_pairs)

    aa_to_int = {aa: i for (i, aa) in enumerate(amino_acids)}
    aa_pair_to_int = {
        aa_pair: i for (i, aa_pair) in enumerate(pairs_of_amino_acids)
    }

    def one_hot_single_site_observation(aa: str):
        if aa not in amino_acids:
            return np.ones(shape=(len(amino_acids), 1))
        res = np.zeros(shape=(len(amino_acids), 1))
        res[aa_to_int[aa]] = 1.0
        return res

    def one_hot_pair_site_observation(aa1: str, aa2: str):
        if aa1 not in amino_acids and aa2 not in amino_acids:
            return np.ones(shape=(len(pairs_of_amino_acids), 1))
        res = np.zeros(shape=(len(pairs_of_amino_acids), 1))
        if aa2 not in amino_acids:
            aa1_id = aa_to_int[aa1]
            for i in range(len(amino_acids)):
                res[aa1_id * len(amino_acids) + i] = 1.0
        if aa1 not in amino_acids:
            aa2_id = aa_to_int[aa2]
            for i in range(len(amino_acids)):
                res[aa2_id + i * len(amino_acids)] = 1.0
        if aa1 in amino_acids and aa2 in amino_acids:
            res[aa_pair_to_int[aa1 + aa2]] = 1.0
        return res

    def populate_leaf_observation_arrays():
        for leaf in tree.leaves():
            seq = msa[leaf]

            single_site_obs = np.zeros(
                shape=(n_independent_sites, len(amino_acids), 1)
            )
            for (i, site_id) in enumerate(independent_sites):
                single_site_obs[i, :, :] = one_hot_single_site_observation(
                    seq[site_id]
                )
            node_observations_single_site[leaf] = single_site_obs

            pair_site_obs = np.zeros(
                shape=(n_contacting_pairs, len(pairs_of_amino_acids), 1)
            )
            for (i, (site_id_1, site_id_2)) in enumerate(contacting_pairs):
                pair_site_obs[i, :, :] = one_hot_pair_site_observation(
                    seq[site_id_1], seq[site_id_2]
                )
            node_observations_pair_site[leaf] = pair_site_obs

    populate_leaf_observation_arrays()

    def populate_internal_node_observation_arrays():
        for node in tree.internal_nodes():
            node_observations_single_site[node] = np.ones(
                shape=(n_independent_sites, len(amino_acids), 1)
            )
            node_observations_pair_site[node] = np.ones(
                shape=(n_contacting_pairs, len(pairs_of_amino_acids), 1)
            )

    populate_internal_node_observation_arrays()
    print(f"Time populate internal node obs: {time.time() - st}")

    st_all_expms = time.time()
    single_site_transition_mats = {}
    pair_site_transition_mats = {}
    # Strategy: compute all matrix exponentials up front with a 3D matrix stack.

    def populate_transition_mats():
        non_root_nodes = [
            node for node in tree.nodes() if not tree.is_root(node)
        ]
        unique_site_rates = sorted(list(set(site_rates)))
        num_cats = len(unique_site_rates)
        site_rate_to_cat = {
            site_rate: cat for (cat, site_rate) in enumerate(unique_site_rates)
        }

        st = time.time()
        if n_independent_sites > 0:
            exponents = []
            for (i, node) in enumerate(non_root_nodes):
                (_, length) = tree.parent(node)
                for (j, site_rate) in enumerate(unique_site_rates):
                    exponents.append(length * site_rate)

            expTQ_1 = matrix_exponential(
                exponents,
                Q_1,
                fact_1,
                reversible_1,
                device_1,
            )

            for (i, node) in enumerate(non_root_nodes):
                single_site_transition_mats_node = np.zeros(
                    shape=(
                        n_independent_sites,
                        len(amino_acids),
                        len(amino_acids),
                    )
                )
                for (j, site_id) in enumerate(independent_sites):
                    single_site_transition_mats_node[j, :, :] = expTQ_1[
                        (i * num_cats) + site_rate_to_cat[site_rates[site_id]],
                        :,
                        :,
                    ]
                single_site_transition_mats[
                    node
                ] = single_site_transition_mats_node
        print(f"\tTime for single-site expms: {time.time() - st}")

        st = time.time()
        if n_contacting_pairs > 0:
            exponents = []
            for (i, node) in enumerate(non_root_nodes):
                (_, length) = tree.parent(node)
                exponents.append(length)

            expTQ_2 = matrix_exponential(
                exponents,
                Q_2,
                fact_2,
                reversible_2,
                device_2,
            )

            for (i, node) in enumerate(non_root_nodes):
                pair_site_transition_mats[node] = expTQ_2[i, :, :][None, :, :]

        print(f"\tTime for pair-site expms: {time.time() - st}")

    populate_transition_mats()
    print(f"Time to populate_transition_mats: {time.time() - st_all_expms}")
    # assert(False)

    dp_single_site = {}
    dp_pair_site = {}

    st = time.time()
    for node in tree.postorder_traversal():
        dp_single_site[node] = np.zeros(
            shape=(n_independent_sites, len(amino_acids), 1)
        )
        dp_pair_site[node] = np.zeros(
            shape=(n_contacting_pairs, len(pairs_of_amino_acids), 1)
        )
        if tree.is_leaf(node):
            continue
        if n_independent_sites > 0:
            for (child, _) in tree.children(node):
                dp_single_site_child = dp_single_site[child]
                max_ll_single_site_child = dp_single_site_child.max(
                    axis=1, keepdims=True
                )
                dp_single_site_child -= max_ll_single_site_child

                log_arg = single_site_transition_mats[child] @ (
                    np.exp(dp_single_site_child)
                    * node_observations_single_site[child]
                )
                log_arg[log_arg < 0] = 0.0

                dp_single_site[node] += (
                    np.log(log_arg) + max_ll_single_site_child
                )
        if n_contacting_pairs > 0:
            for (child, _) in tree.children(node):
                dp_pair_site_child = dp_pair_site[child]
                max_ll_pair_site_child = dp_pair_site_child.max(
                    axis=1, keepdims=True
                )
                dp_pair_site_child -= max_ll_pair_site_child

                log_arg = pair_site_transition_mats[child] @ (
                    np.exp(dp_pair_site_child)
                    * node_observations_pair_site[child]
                )
                log_arg[log_arg < 0] = 0.0

                dp_pair_site[node] += np.log(log_arg) + max_ll_pair_site_child

    if n_independent_sites > 0:
        dp_single_site_root = dp_single_site[tree.root()]
        max_ll_single_site_root = dp_single_site_root.max(axis=1, keepdims=True)
        dp_single_site_root -= max_ll_single_site_root

        log_arg = pi_1.reshape(1, 1, -1) @ (
            np.exp(dp_single_site_root)
            * node_observations_single_site[tree.root()]
        )
        log_arg[log_arg < 0] = 0.0

        res_single_site = np.log(log_arg) + max_ll_single_site_root

    if n_contacting_pairs > 0:
        dp_pair_site_root = dp_pair_site[tree.root()]
        max_ll_pair_site_root = dp_pair_site_root.max(axis=1, keepdims=True)
        dp_pair_site_root -= max_ll_pair_site_root

        log_arg = pi_2.reshape(1, 1, -1) @ (
            np.exp(dp_pair_site_root) * node_observations_pair_site[tree.root()]
        )
        log_arg[log_arg < 0] = 0.0

        res_pair_site = np.log(log_arg) + max_ll_pair_site_root

    lls = [0] * num_sites
    if n_independent_sites > 0:
        for (i, site_id) in enumerate(independent_sites):
            lls[site_id] = res_single_site[i, 0, 0]
    if n_contacting_pairs > 0:
        for (i, (site_id_1, site_id_2)) in enumerate(contacting_pairs):
            lls[site_id_1] = res_pair_site[i, 0, 0] / 2.0
            lls[site_id_2] = res_pair_site[i, 0, 0] / 2.0

    print(f"Time for dp = {time.time() - st}")
    print(f"Total time = {time.time() - st_all}")
    # assert(False)  # Uncomment to see timing results when running tests
    return sum(lls), lls


def compute_log_likelihoods(
    tree_dir: str,
    msa_dir: str,
    site_rates_dir: str,
    contact_map_dir: str,
    families: List[str],
    amino_acids: List[str],
    pi_1_path: str,
    Q_1_path: str,
    reversible_1: bool,
    device_1: str,
    pi_2_path: str,
    Q_2_path: str,
    reversible_2: bool,
    device_2: str,
    output_likelihood_dir: str,
    num_processes: int,
    use_cpp_implementation: bool = False,
) -> None:
    """
    Compute log-likelihoods under the given model.

    Given trees, MSAs, site rates, contact maps, and models for the evolution
    of contacting sites and non-contacting sites, the log-likelihood of each
    tree is computed, at the resolution of single-sites and contacting pairs.
    The model is given by the state distribution of the root node, and the rate
    matrix which describes the evolution of states.

    Details:
    - For each position, it must be either in contact with exactly 1 other
        position, or not be in contact with any other position. The diagonal
        of the contact matrix is ignored.
    - The Q_2 matrix is sparse: only 2 * len(amino_acids) - 1 entries in each
        row are non-zero, since only one amino acid in a contacting pair
        can mutate at a time.

    Args:
        tree_dir: Directory to the trees stored in friendly format.
        msa_dir: Directory to the multiple sequence alignments in FASTA format.
        site_rates_dir: Directory to the files containing the rates at which
            each site evolves. Rates for sites that co-evolve are ignored.
        contact_map_dir: Directory to the contact maps stored as
            space-separated binary matrices.
        families: The protein families for which to perform the computation.
        amino_acids: The list of amino acids.
        pi_1_path: Path to an array of length len(amino_acids). It indicates,
            for sites that evolve independently (i.e. that are not in contact
            with any other site), the probabilities for the root state.
        Q_1_path: Path to an array of size len(amino_acids) x len(amino_acids),
            the rate matrix describing the evolution of sites that evolve
            independently (i.e. that are not in contact with any other site).
        reversible_1: Whether to use reversibility of Q_1 to compute the
            matrix exponential faster.
        device_1: Whether to use 'cpu' or 'cuda' to compute the matrix
            exponentials of Q_1.
        pi_2_path: Path to an array of length len(amino_acids) ** 2. It
            indicates, for sites that are in contact, the probabilities for
            their root state.
        Q_2_path: Path to an array of size (len(amino_acids) ** 2) x
            (len(amino_acids) ** 2), the rate matrix describing the evolution
            of sites that are in contact.
        reversible_2: Whether to use reversibility of Q_2 to compute the
            matrix exponential faster.
        device_2: Whether to use 'cpu' or 'cuda' to compute the matrix
            exponentials of Q_2.
        output_likelihood_dir: Directory where to write the log-likelihoods,
            with site-level resolution.
        num_processes: Number of processes used to parallelize computation.
        use_cpp_implementation: If to use efficient C++ implementation
            instead of Python.
    """
    if use_cpp_implementation:
        raise NotImplementedError

    for (
        family
    ) in (
        families
    ):  # I don't use Python multiprocessing bc GPU seems to be the bottleneck. # UPDATE: If the SVD trick is fast enough on 1 cpu, we are back in business! We can use CPU instead of GPU.
        tree_path = os.path.join(tree_dir, family + ".txt")
        msa_path = os.path.join(msa_dir, family + ".txt")
        site_rates_path = os.path.join(site_rates_dir, family + ".txt")
        contact_map_path = os.path.join(contact_map_dir, family + ".txt")

        tree = read_tree(tree_path)
        msa = read_msa(msa_path)
        site_rates = read_site_rates(site_rates_path)
        contact_map = read_contact_map(contact_map_path)
        pi_1_df = read_probability_distribution(pi_1_path)
        Q_1_df = read_rate_matrix(Q_1_path)
        pi_2_df = read_probability_distribution(pi_2_path)
        Q_2_df = read_rate_matrix(Q_2_path)

        pairs_of_amino_acids = [
            aa1 + aa2 for aa1 in amino_acids for aa2 in amino_acids
        ]
        # Validate states of rate matrices and root distribution
        if list(pi_1_df.index) != amino_acids:
            raise Exception(
                f"pi_1 index is:\n{list(pi_1_df.index)}\nbut expected amino "
                f"acids:\n{amino_acids}"
            )
        if list(pi_2_df.index) != pairs_of_amino_acids:
            raise Exception(
                f"pi_2 index is:\n{list(pi_2_df.index)}\nbut expected pairs of "
                f"amino acids:\n{pairs_of_amino_acids}"
            )
        if list(Q_1_df.index) != amino_acids:
            raise Exception(
                f"Q_1 index is:\n{list(Q_1_df.index)}\n\nbut expected amino "
                f"acids:\n{amino_acids}"
            )
        if list(Q_1_df.columns) != amino_acids:
            raise Exception(
                f"Q_1 columns are:\n{list(Q_1_df.columns)}\n\nbut expected "
                f"amino acids:\n{amino_acids}"
            )
        if list(Q_2_df.index) != pairs_of_amino_acids:
            raise Exception(
                f"Q_2 index is:\n{list(Q_2_df.index)}\n\nbut expected pairs of "
                f"amino acids:\n{pairs_of_amino_acids}"
            )
        if list(Q_2_df.columns) != pairs_of_amino_acids:
            raise Exception(
                f"Q_1 columns are:\n{list(Q_2_df.columns)}\n\nbut expected "
                f"pairs of amino acids:\n{pairs_of_amino_acids}"
            )

        fact_1 = (
            FactorizedReversibleModel(Q_1_df.to_numpy())
            if reversible_1
            else None
        )
        fact_2 = (
            FactorizedReversibleModel(Q_2_df.to_numpy())
            if reversible_1
            else None
        )

        ll, lls = dp_likelihood_computation(
            tree=tree,
            msa=msa,
            contact_map=contact_map,
            site_rates=site_rates,
            amino_acids=amino_acids,
            pi_1=pi_1_df.to_numpy(),
            Q_1=Q_1_df.to_numpy(),
            fact_1=fact_1,
            reversible_1=reversible_1,
            device_1=device_1,
            pi_2=pi_2_df.to_numpy(),
            Q_2=Q_2_df.to_numpy(),
            fact_2=fact_2,
            reversible_2=reversible_2,
            device_2=device_2,
        )
        ll_path = os.path.join(output_likelihood_dir, family + ".txt")
        write_log_likelihood((ll, lls), ll_path)
