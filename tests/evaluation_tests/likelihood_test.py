import os
import tempfile
import unittest
from typing import Dict, List, Tuple

import itertools
import numpy as np
import pandas as pd
import pytest
from parameterized import parameterized

from src.counting import count_co_transitions
from src.counting import count_transitions
from src.io import read_count_matrices
from src.evaluation import compute_log_likelihoods
from tests.utils import create_synthetic_contact_map
from src.markov_chain import matrix_exponential, wag_matrix, wag_stationary_distribution, chain_product, compute_stationary_distribution

from src.io import Tree
import src


def log_sum_exp(lls: np.array) -> float:
    m = lls.max()
    lls -= m
    res = np.log(np.sum(np.exp(lls))) + m
    return res


def create_fake_msa_and_contact_map_and_site_rates(
    tree: Tree,
    amino_acids: List[str],
    random_seed: int,
    num_rate_categories: int,
) -> Tuple[Dict[str, str], np.array, List[float]]:
    """
    Create fake data for a tree.
    """
    np.random.seed(random_seed)

    num_leaves = sum([tree.is_leaf(v) for v in tree.nodes()])
    single_site_patterns = [''.join(pattern) for pattern in list(itertools.product(amino_acids, repeat=num_leaves))]
    pair_of_site_patterns = list(itertools.product(single_site_patterns, repeat=2))
    # print(f"single_site_patterns = {single_site_patterns}")
    # print(f"pair_of_site_patterns = {pair_of_site_patterns}")
    num_sites = len(single_site_patterns) + 2 * len(pair_of_site_patterns)
    contact_map = create_synthetic_contact_map(
        num_sites=num_sites,
        num_sites_in_contact=2 * len(pair_of_site_patterns),
        random_seed=random_seed
    )
    contacting_pairs = list(zip(*np.where(contact_map == 1)))
    np.random.shuffle(contacting_pairs)
    contacting_pairs = [(i, j) for (i, j) in contacting_pairs if i < j]
    contacting_sites = list(sum(contacting_pairs, ()))
    independent_sites = [
        i for i in range(num_sites) if i not in contacting_sites
    ]
    np.random.shuffle(independent_sites)
    # print(f"contacting_pairs = {contacting_pairs}")
    # print(f"independent_sites = {independent_sites}")

    msa_array = np.zeros(shape=(num_sites, num_leaves), dtype=str)
    for i, site_idx in enumerate(independent_sites):
        for leaf_idx in range(num_leaves):
            msa_array[site_idx, leaf_idx] = single_site_patterns[i][leaf_idx]
    for i, (site_idx_1, site_idx_2) in enumerate(contacting_pairs):
        for leaf_idx in range(num_leaves):
            msa_array[site_idx_1, leaf_idx] = pair_of_site_patterns[i][0][leaf_idx]
            msa_array[site_idx_2, leaf_idx] = pair_of_site_patterns[i][1][leaf_idx]
    # print(f"msa_array = {msa_array}")
    # for i in range(num_sites):
    #     print(i, msa_array[i])
    msa = {
        leaf: ''.join(msa_array[:, i]) for i, leaf in enumerate(tree.leaves())
    }
    # print(f"msa = {msa}")
    site_rates = [0.5 * np.log(2 + i) for i in range(num_rate_categories)] * (int(num_sites / num_rate_categories) + 1)
    site_rates = site_rates[:num_sites]
    np.random.shuffle(site_rates)
    # print(f"site_rates = {site_rates}")
    return msa, contact_map, site_rates


def brute_force_likelihood_computation(
    tree: Tree,
    msa: Dict[str, str],
    contact_map: np.array,
    site_rates: List[float],
    amino_acids: List[str],
    pi_1: np.array,
    Q_1: np.array,
    pi_2: np.array,
    Q_2: np.array,
) -> Tuple[float, List[float]]:
    """
    Compute data loglikelihood by brute force.

    The ancestral states are marginalized out by hand.
    """
    num_sites = contact_map.shape[0]

    contacting_pairs = list(zip(*np.where(contact_map == 1)))
    contacting_pairs = [(i, j) for (i, j) in contacting_pairs if i < j]
    contacting_sites = list(sum(contacting_pairs, ()))
    independent_sites = [
        i for i in range(num_sites) if i not in contacting_sites
    ]

    pairs_of_amino_acids = [
        aa1 + aa2 for aa1 in amino_acids for aa2 in amino_acids
    ]
    aa_to_int = {aa: i for (i, aa) in enumerate(amino_acids)}
    aa_to_int['-'] = slice(0, len(amino_acids), 1)  # Slice yay!
    aa_pair_to_int = {
        aa_pair: i for (i, aa_pair) in enumerate(pairs_of_amino_acids)
    }

    num_internal_nodes = sum([not tree.is_leaf(v) for v in tree.nodes()])
    single_site_patterns = [''.join(pattern) for pattern in itertools.product(amino_acids, repeat=num_internal_nodes)]
    # print(f"single_site_patterns = {single_site_patterns}")
    pair_of_site_patterns = list(itertools.product(single_site_patterns, repeat=2))
    # print(f"pair_of_site_patterns = {pair_of_site_patterns}")

    # Compute node to int mapping
    num_nodes = len(tree.nodes())
    node_to_int = {}
    int_to_node = [-1] * num_nodes
    for i, node in enumerate(tree.internal_nodes()):
        node_to_int[node] = i
        int_to_node[i] = node
    num_internal_nodes = len(tree.internal_nodes())
    for i, node in enumerate(tree.leaves()):
        node_to_int[node] = num_internal_nodes + i
        int_to_node[num_internal_nodes + i] = node

    lls = [-1] * num_sites
    for site_idx in independent_sites:
        # Pre-compute matrix exponentials
        expms = {}
        for node in tree.nodes():
            if not tree.is_root(node):
                (parent, branch_length) = tree.parent(node)
                expms[node] = matrix_exponential(branch_length * Q_1 * site_rates[site_idx])

        # Compute the likelihood of independent site i
        lls_for_patterns = []
        for anc_states in single_site_patterns:
            leaf_states = ''.join([msa[int_to_node[j]][site_idx] for j in range(num_internal_nodes, num_nodes, 1)])
            all_states = list(anc_states + leaf_states)
            # Compute ll of this pattern
            ll_joint = 0
            ll_joint += np.log(pi_1[aa_to_int[all_states[node_to_int[tree.root()]]]])  # root likelihood
            for (u, v, _) in tree.edges():
                ll_joint += np.log(
                    np.sum(
                        expms[v][
                            aa_to_int[all_states[node_to_int[u]]],
                            aa_to_int[all_states[node_to_int[v]]],
                        ]
                    )
                )
            lls_for_patterns.append(ll_joint)
        lls[site_idx] = log_sum_exp(np.array(lls_for_patterns))
    for (site_idx_1, site_idx_2) in contacting_pairs:
        lls[site_idx_1] = -1
        lls[site_idx_2] = -1
    return sum(lls), lls


class TestComputeLogLikelihoods(unittest.TestCase):
    def test_small_wag_3_seqs(self):
        """
        This was manually verified with FastTree.
        """
        tree = Tree()
        tree.add_nodes(["r", "l1", "l2", "l3"])
        tree.add_edges(
            [
                ("r", "l1", 0.0),
                ("r", "l2", 1.120547166),
                ("r", "l3", 3.402392896),
            ]
        )
        msa = {
            'l1': 'S',
            'l2': 'T',
            'l3': 'G',
        }
        contact_map = np.eye(1)
        site_rates = [1.0]
        ll, lls = brute_force_likelihood_computation(
            tree=tree,
            msa=msa,
            contact_map=contact_map,
            site_rates=site_rates,
            amino_acids=src.utils.amino_acids,
            pi_1=wag_stationary_distribution().to_numpy(),
            Q_1=wag_matrix().to_numpy(),
            pi_2=None,
            Q_2=None,
        )
        # TODO: Test actual Python implementation too!
        np.testing.assert_almost_equal(ll, -7.343870, decimal=4)
        np.testing.assert_almost_equal(lls, [-7.343870], decimal=4)

    def test_small_wag_4_seqs_1_site(self):
        """
        This was manually verified with FastTree.
        """
        tree = Tree()
        tree.add_nodes(["r", "i1", "l1", "l2", "l3", "l4"])
        tree.add_edges(
            [
                ("r", "l1", 0.0),
                ("r", "l2", 1.121352212),
                ("r", "i1", 1.840784231),
                ("i1", "l3", 1.870540996),
                ("i1", "l4", 2.678783814),
            ]
        )
        msa = {
            'l1': 'S',
            'l2': 'T',
            'l3': 'G',
            'l4': 'D'
        }
        contact_map = np.eye(1)
        site_rates = [1.0]
        ll, lls = brute_force_likelihood_computation(
            tree=tree,
            msa=msa,
            contact_map=contact_map,
            site_rates=site_rates,
            amino_acids=src.utils.amino_acids,
            pi_1=wag_stationary_distribution().to_numpy(),
            Q_1=wag_matrix().to_numpy(),
            pi_2=None,
            Q_2=None,
        )
        # TODO: test actual Python implementation!
        np.testing.assert_almost_equal(ll, -10.091868, decimal=4)
        np.testing.assert_almost_equal(lls, [-10.091868], decimal=4)

    def test_small_wag_4_seqs_2_sites_and_gaps(self):
        """
        This was manually verified with FastTree.
        """
        tree = Tree()
        tree.add_nodes(["r", "i1", "l1", "l2", "l3", "l4"])
        tree.add_edges(
            [
                ("r", "l1", 0.0),
                ("r", "l2", 1.121562482),
                ("r", "i1", 1.719057732),
                ("i1", "l3", 1.843908633),
                ("i1", "l4", 2.740236263),
            ]
        )
        msa = {
            'l1': 'SS',
            'l2': 'TT',
            'l3': 'GG',
            'l4': 'D-'
        }
        contact_map = np.eye(2)
        site_rates = [1.0, 1.0]
        ll, lls = brute_force_likelihood_computation(
            tree=tree,
            msa=msa,
            contact_map=contact_map,
            site_rates=site_rates,
            amino_acids=src.utils.amino_acids,
            pi_1=wag_stationary_distribution().to_numpy(),
            Q_1=wag_matrix().to_numpy(),
            pi_2=None,
            Q_2=None,
        )
        # TODO: test actual Python implementation!
        np.testing.assert_almost_equal(ll, -17.436349, decimal=4)
        np.testing.assert_almost_equal(
            lls, [-10.092142, -7.344207], decimal=4
        )

    def test_small_wag_x_wag_3_seqs(self):
        """
        This was manually verified with FastTree.
        """
        tree = Tree()
        tree.add_nodes(["r", "l1", "l2", "l3"])
        tree.add_edges(
            [
                ("r", "l1", 0.0),
                ("r", "l2", 1.120547166),
                ("r", "l3", 3.402392896),
            ]
        )
        msa = {
            'l1': 'SK',
            'l2': 'TI',
            'l3': 'GL',
        }
        contact_map = np.ones((2, 2))
        site_rates = [1.0, 1.0]
        wag = wag_matrix().to_numpy()
        wag_x_wag = chain_product(wag)
        pi_x_pi = compute_stationary_distribution(wag_x_wag)
        ll, lls = brute_force_likelihood_computation(
            tree=tree,
            msa=msa,
            contact_map=contact_map,
            site_rates=site_rates,
            amino_acids=src.utils.amino_acids,
            pi_1=wag_stationary_distribution().to_numpy(),
            Q_1=wag,
            pi_2=pi_x_pi,
            Q_2=wag_x_wag,
        )
        # TODO: Test actual Python implementation too!
        np.testing.assert_almost_equal(ll, -7.343870 + -9.714873, decimal=4)
        np.testing.assert_almost_equal(lls, [-7.343870, -9.714873], decimal=4)

    # @parameterized.expand(
    #     [("3 processes", 3)]
    # )
    # def test_2(self, name, num_processes):
    #     tree = Tree()
    #     tree.add_nodes(["r", "i1", "l1", "l2", "l3"])
    #     tree.add_edges(
    #         [
    #             ("r", "i1", 0.14),
    #             ("i1", "l1", 1.14),
    #             ("i1", "l2", 0.71),
    #             ("r", "l3", 3.14),
    #         ]
    #     )
    #     amino_acids = ["G", "P"]
    #     msa, contact_map, site_rates = \
    #         create_fake_msa_and_contact_map_and_site_rates(
    #             tree=tree,
    #             amino_acids=amino_acids,
    #             random_seed=1,
    #             num_rate_categories=3
    #         )

    #     ll, lls = brute_force_likelihood_computation(
    #         tree=tree,
    #         msa=msa,
    #         contact_map=contact_map,
    #         site_rates=site_rates,
    #         amino_acids=amino_acids,
    #         pi_1=np.array(
    #             [0.75, 0.25]
    #         ),
    #         Q_1=np.array(
    #             [
    #                 [-1, 1],
    #                 [3, -3]
    #             ],
    #             dtype=float
    #         ),
    #         pi_2=None,
    #         Q_2=None,
    #     )
    #     print(f"lls = {lls}")

    #     assert(False)

    # def test_1(self):
    #     tree = Tree()
    #     tree.add_nodes(["r", "i1", "l1", "l2"])
    #     tree.add_edges(
    #         [
    #             ("r", "i1", 0.14),
    #             ("i1", "l1", 1.14),
    #             ("i1", "l2", 0.71),
    #         ]
    #     )
    #     msa, contact_map, site_rates = \
    #         create_fake_msa_and_contact_map_and_site_rates(
    #             tree=tree,
    #             amino_acids=["G", "P"],
    #             random_seed=1
    #         )
    #     assert(False)
