"""
TODO:
/ ABORTED (too hard; I will just do some large test using whatever tree FastTree likes to estimate to test my Python Felsenstein Implementation's total LL. Then, I will just rely on comparisson against brute force method for smaller trees with site-level resolution):
    Running FastTree with a fixed Tree:
    Need to try to turn off branch length estimation with a fixed topology... Apparently line 2331
    LL gets printed in line 2418. 2423 writes to the provided logfile.
    Alignment gets read and printed in 2089.
    All the interesting stuff in main seems to happen in 2128 if statement, the alternative to 2110 which happens when make_matrix=false (always the case unless -makematrix is specified as a command line arg)
    2188 (2193) reads the input tree if provided
"""
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
from src.io import write_tree, write_contact_map, write_msa, write_site_rates, write_probability_distribution, write_rate_matrix, read_log_likelihood, read_tree, read_msa, read_site_rates, read_contact_map
from src.evaluation import compute_log_likelihoods
from tests.utils import create_synthetic_contact_map
from src.markov_chain import matrix_exponential, wag_matrix, wag_stationary_distribution, chain_product, compute_stationary_distribution,\
    equ_matrix
from src.evaluation import compute_log_likelihoods, brute_force_likelihood_computation

from src.io import Tree
import src

DATA_DIR = "./tests/evaluation_tests/test_input_data"


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


def likelihood_computation_wrapper(
    tree: Tree,
    msa: Dict[str, str],
    contact_map: np.array,
    site_rates: List[float],
    amino_acids: List[str],
    pi_1: np.array,
    Q_1: np.array,
    pi_2: np.array,
    Q_2: np.array,
    method: str,
) -> Tuple[float, List[float]]:
    """
    Compute data loglikelihood by one of several methods
    """
    if method == "brute_force":
        return brute_force_likelihood_computation(
            tree=tree,
            msa=msa,
            contact_map=contact_map,
            site_rates=site_rates,
            amino_acids=amino_acids,
            pi_1=pi_1,
            Q_1=Q_1,
            pi_2=pi_2,
            Q_2=Q_2,
        )
    elif method == "python" or method == "C++":
        family = "fam1"
        families = [family]
        use_cpp_implementation = method == "C++"
        with tempfile.TemporaryDirectory() as tree_dir:
            tree_path = os.path.join(tree_dir, family + ".txt")
            write_tree(tree, tree_path)
            with tempfile.TemporaryDirectory() as msa_dir:
                msa_path = os.path.join(msa_dir, family + ".txt")
                write_msa(msa, msa_path)
                with tempfile.TemporaryDirectory() as contact_map_dir:
                    contact_map_path = os.path.join(contact_map_dir, family + ".txt")
                    write_contact_map(contact_map, contact_map_path)
                    with tempfile.TemporaryDirectory() as site_rates_dir:
                        site_rates_path = os.path.join(site_rates_dir, family + ".txt")
                        write_site_rates(site_rates, site_rates_path)
                        with tempfile.NamedTemporaryFile("w") as pi_1_file:
                            pi_1_path = pi_1_file.name
                            # pi_1_path = "./pi_1_path.txt"
                            write_probability_distribution(pi_1, amino_acids, pi_1_path)
                            with tempfile.NamedTemporaryFile("w") as Q_1_file:
                                Q_1_path = Q_1_file.name
                                # Q_1_path = "./Q_1_path.txt"
                                write_rate_matrix(Q_1, amino_acids, Q_1_path)
                                with tempfile.NamedTemporaryFile("w") as pi_2_file:
                                    pi_2_path = pi_2_file.name
                                    # pi_2_path = "./pi_2_path.txt"
                                    amino_acid_pairs = [
                                        aa1 + aa2 for aa1 in amino_acids for aa2 in amino_acids
                                    ]
                                    write_probability_distribution(pi_2,  amino_acid_pairs, pi_2_path)
                                    with tempfile.NamedTemporaryFile("w") as Q_2_file:
                                        Q_2_path = Q_2_file.name
                                        # Q_2_path = "./Q_2_path.txt"
                                        write_rate_matrix(Q_2,  amino_acid_pairs, Q_2_path)
                                        with tempfile.TemporaryDirectory() as log_likelihood_dir:
                                            # log_likelihood_dir = "log_likelihood_dir"
                                            compute_log_likelihoods(
                                                tree_dir=tree_dir,
                                                msa_dir=msa_dir,
                                                site_rates_dir=site_rates_dir,
                                                contact_map_dir=contact_map_dir,
                                                families=families,
                                                amino_acids=amino_acids,
                                                pi_1_path=pi_1_path,
                                                Q_1_path=Q_1_path,
                                                pi_2_path=pi_2_path,
                                                Q_2_path=Q_2_path,
                                                output_likelihood_dir=log_likelihood_dir,
                                                num_processes=1,
                                                use_cpp_implementation=use_cpp_implementation,
                                            )
                                            log_likelihood_path = os.path.join(
                                                log_likelihood_dir, family + ".txt"
                                            )
                                            ll, lls = read_log_likelihood(
                                                log_likelihood_path
                                            )
                                            return ll, lls
    else:
        raise NotImplementedError(f"Unknown method: {method}")


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
        equ = equ_matrix().to_numpy()
        pi = compute_stationary_distribution(equ)
        equ_x_equ = chain_product(equ, equ)
        pi_x_pi = compute_stationary_distribution(equ_x_equ)
        ll, lls = likelihood_computation_wrapper(
            tree=tree,
            msa=msa,
            contact_map=contact_map,
            site_rates=site_rates,
            amino_acids=src.utils.amino_acids,
            pi_1=wag_stationary_distribution().to_numpy(),
            Q_1=wag_matrix().to_numpy(),
            pi_2=pi_x_pi,
            Q_2=equ_x_equ,
            method="python",
        )
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
        equ = equ_matrix().to_numpy()
        pi = compute_stationary_distribution(equ)
        equ_x_equ = chain_product(equ, equ)
        pi_x_pi = compute_stationary_distribution(equ_x_equ)
        ll, lls = likelihood_computation_wrapper(
            tree=tree,
            msa=msa,
            contact_map=contact_map,
            site_rates=site_rates,
            amino_acids=src.utils.amino_acids,
            pi_1=wag_stationary_distribution().to_numpy(),
            Q_1=wag_matrix().to_numpy(),
            pi_2=pi_x_pi,
            Q_2=equ_x_equ,
            method="python",
        )
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
        equ = equ_matrix().to_numpy()
        equ_x_equ = chain_product(equ, equ)
        pi_x_pi = compute_stationary_distribution(equ_x_equ)
        ll, lls = likelihood_computation_wrapper(
            tree=tree,
            msa=msa,
            contact_map=contact_map,
            site_rates=site_rates,
            amino_acids=src.utils.amino_acids,
            pi_1=wag_stationary_distribution().to_numpy(),
            Q_1=wag_matrix().to_numpy(),
            pi_2=pi_x_pi,
            Q_2=equ_x_equ,
            method="python",
        )
        np.testing.assert_almost_equal(ll, -17.436349, decimal=4)
        np.testing.assert_almost_equal(
            lls, [-10.092142, -7.344207], decimal=4
        )

    def test_small_equ_x_equ_3_seqs(self):
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
        # contact_map = np.eye(2)
        site_rates = [1.0, 1.0]
        equ = equ_matrix().to_numpy()
        pi = compute_stationary_distribution(equ)
        equ_x_equ = chain_product(equ, equ)
        pi_x_pi = compute_stationary_distribution(equ_x_equ)
        np.testing.assert_almost_equal(
            matrix_exponential(equ_x_equ)[0, 0],
            matrix_exponential(equ)[0, 0] ** 2
        )
        ll, lls = likelihood_computation_wrapper(
            tree=tree,
            msa=msa,
            contact_map=contact_map,
            site_rates=site_rates,
            amino_acids=src.utils.amino_acids,
            pi_1=pi,
            Q_1=equ,
            pi_2=pi_x_pi,
            Q_2=equ_x_equ,
            method="python",
        )
        np.testing.assert_almost_equal(ll, -9.382765 * 2, decimal=4)
        np.testing.assert_almost_equal(lls, [-9.382765, -9.382765], decimal=4)

    def test_small_equ_x_wag_3_seqs(self):
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
        # contact_map = np.eye(2)
        site_rates = [1.0, 1.0]
        equ = equ_matrix().to_numpy()
        wag = wag_matrix().to_numpy()
        equ_x_wag = chain_product(equ, wag)
        np.testing.assert_almost_equal(
            matrix_exponential(equ_x_wag)[0, 0],
            matrix_exponential(equ)[0, 0] * matrix_exponential(wag)[0, 0]
        )
        pi_2 = compute_stationary_distribution(equ_x_wag)
        pi = compute_stationary_distribution(equ)
        ll, lls = likelihood_computation_wrapper(
            tree=tree,
            msa=msa,
            contact_map=contact_map,
            site_rates=site_rates,
            amino_acids=src.utils.amino_acids,
            pi_1=pi,
            Q_1=equ,
            pi_2=pi_2,
            Q_2=equ_x_wag,
            method="python",
        )
        epected_ll = -9.382765 + -9.714873
        np.testing.assert_almost_equal(ll, epected_ll, decimal=4)
        np.testing.assert_almost_equal(lls, [epected_ll / 2, epected_ll / 2], decimal=4)

    def test_small_wag_x_equ_3_seqs(self):
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
            'l1': 'KS',
            'l2': 'IT',
            'l3': 'LG',
        }
        contact_map = np.ones((2, 2))
        # contact_map = np.eye(2)
        site_rates = [1.0, 1.0]
        equ = equ_matrix().to_numpy()
        wag = wag_matrix().to_numpy()
        wag_x_equ = chain_product(wag, equ)
        pi_2 = compute_stationary_distribution(wag_x_equ)
        np.testing.assert_almost_equal(
            matrix_exponential(wag_x_equ)[0, 0],
            matrix_exponential(wag)[0, 0] * matrix_exponential(equ)[0, 0]
        )
        pi = compute_stationary_distribution(equ)
        ll, lls = likelihood_computation_wrapper(
            tree=tree,
            msa=msa,
            contact_map=contact_map,
            site_rates=site_rates,
            amino_acids=src.utils.amino_acids,
            pi_1=pi,
            Q_1=equ,
            pi_2=pi_2,
            Q_2=wag_x_equ,
            method="python",
        )
        epected_ll = -9.714873 + -9.382765
        np.testing.assert_almost_equal(ll, epected_ll, decimal=4)
        np.testing.assert_almost_equal(lls, [epected_ll / 2, epected_ll / 2], decimal=4)

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
        # contact_map = np.eye(2)
        site_rates = [1.0, 1.0]
        wag = wag_matrix().to_numpy()
        pi = compute_stationary_distribution(wag)
        wag_x_wag = chain_product(wag, wag)
        np.testing.assert_almost_equal(
            matrix_exponential(wag_x_wag)[0, 0],
            matrix_exponential(wag)[0, 0] ** 2
        )
        pi_x_pi = compute_stationary_distribution(wag_x_wag)
        ll, lls = likelihood_computation_wrapper(
            tree=tree,
            msa=msa,
            contact_map=contact_map,
            site_rates=site_rates,
            amino_acids=src.utils.amino_acids,
            pi_1=pi,
            Q_1=wag,
            pi_2=pi_x_pi,
            Q_2=wag_x_wag,
            method="python",
        )
        ll_expected = -7.343870 + -9.714873
        np.testing.assert_almost_equal(ll, ll_expected, decimal=4)
        np.testing.assert_almost_equal(lls, [ll_expected / 2, ll_expected / 2], decimal=4)

    def test_small_wag_x_wag_3_seqs_many_sites(self):
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
            'l1': 'KSRMFCP',
            'l2': 'ITVDQAE',
            'l3': 'LGYNGHW',
        }
        contact_map = np.array(
            [
                [1, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 1, 0],
                [0, 0, 1, 1, 0, 0, 0],
                [0, 0, 1, 1, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0],
                [0, 1, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 1],
            ]
        )
        site_rates = [1.0, 2.0, 2.0, 1.0, 1.0, 2.0, 1.0]  # I use 2s to make sure site rates are not getting used for coevolution
        wag = wag_matrix().to_numpy()
        pi = compute_stationary_distribution(wag)
        wag_x_wag = chain_product(wag, wag)
        np.testing.assert_almost_equal(
            matrix_exponential(wag_x_wag)[0, 0],
            matrix_exponential(wag)[0, 0] ** 2
        )
        pi_x_pi = compute_stationary_distribution(wag_x_wag)
        ll, lls = likelihood_computation_wrapper(
            tree=tree,
            msa=msa,
            contact_map=contact_map,
            site_rates=site_rates,
            amino_acids=src.utils.amino_acids,
            pi_1=pi,
            Q_1=wag,
            pi_2=pi_x_pi,
            Q_2=wag_x_wag,
            method="python",
        )
        lls_expected = [
            -9.714873,
            (-7.343870 + -10.78960) / 2,
            (-10.56782 + -11.85804) / 2,
            (-11.85804 + -10.56782) / 2,
            -11.38148,
            (-10.78960 + -7.343870) / 2,
            -11.31551,
        ]
        np.testing.assert_almost_equal(lls, lls_expected, decimal=4)
        np.testing.assert_almost_equal(ll, sum(lls_expected), decimal=4)

    def test_small_wag_x_wag_3_seqs_many_sites_and_gaps(self):
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
            'l1': '--RMF--',
            'l2': '-TV-QAE',
            'l3': '-G--G-W',
        }
        contact_map = np.array(
            [
                [1, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 1, 0],
                [0, 0, 1, 1, 0, 0, 0],
                [0, 0, 1, 1, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0],
                [0, 1, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 1],
            ]
        )
        # contact_map = np.eye(7)
        site_rates = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]  # I use 2s to make sure site rates are not getting used for coevolution
        wag = wag_matrix().to_numpy()
        pi = compute_stationary_distribution(wag)
        wag_x_wag = chain_product(wag, wag)
        np.testing.assert_almost_equal(
            matrix_exponential(wag_x_wag)[0, 0],
            matrix_exponential(wag)[0, 0] ** 2
        )
        pi_x_pi = compute_stationary_distribution(wag_x_wag)
        ll, lls = likelihood_computation_wrapper(
            tree=tree,
            msa=msa,
            contact_map=contact_map,
            site_rates=site_rates,
            amino_acids=src.utils.amino_acids,
            pi_1=pi,
            Q_1=wag,
            pi_2=pi_x_pi,
            Q_2=wag_x_wag,
            method="python",
        )
        lls_expected = [
            0.0,
            (-5.323960 + -2.446133) / 2,
            (-6.953994 + -3.937202) / 2,
            (-3.937202 + -6.953994) / 2,
            -11.38148,
            (-2.446133 + -5.323960) / 2,
            -7.469626,
        ]
        np.testing.assert_almost_equal(lls, lls_expected, decimal=4)
        np.testing.assert_almost_equal(ll, sum(lls_expected), decimal=4)

    def test_small_wag_x_wag_3_seqs_many_sites(self):
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
            'l1': 'KSRMFCP',
            'l2': 'ITVDQAE',
            'l3': 'LGYNGHW',
        }
        contact_map = np.array(
            [
                [1, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 1, 0],
                [0, 0, 1, 1, 0, 0, 0],
                [0, 0, 1, 1, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0],
                [0, 1, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 1],
            ]
        )
        site_rates = [1.0, 2.0, 2.0, 1.0, 1.0, 2.0, 1.0]  # I use 2s to make sure site rates are not getting used for coevolution
        wag = wag_matrix().to_numpy()
        pi = compute_stationary_distribution(wag)
        wag_x_wag = chain_product(wag, wag)
        np.testing.assert_almost_equal(
            matrix_exponential(wag_x_wag)[0, 0],
            matrix_exponential(wag)[0, 0] ** 2
        )
        pi_x_pi = compute_stationary_distribution(wag_x_wag)
        ll, lls = likelihood_computation_wrapper(
            tree=tree,
            msa=msa,
            contact_map=contact_map,
            site_rates=site_rates,
            amino_acids=src.utils.amino_acids,
            pi_1=pi,
            Q_1=wag,
            pi_2=pi_x_pi,
            Q_2=wag_x_wag,
            method="python",
        )
        lls_expected = [
            -9.714873,
            (-7.343870 + -10.78960) / 2,
            (-10.56782 + -11.85804) / 2,
            (-11.85804 + -10.56782) / 2,
            -11.38148,
            (-10.78960 + -7.343870) / 2,
            -11.31551,
        ]
        np.testing.assert_almost_equal(lls, lls_expected, decimal=4)
        np.testing.assert_almost_equal(ll, sum(lls_expected), decimal=4)

    def test_small_wag_x_wag_2_seqs_many_sites(self):
        """
        This was manually verified with FastTree.
        """
        tree = Tree()
        tree.add_nodes(["r", "i1", "l1", "l2"])
        tree.add_edges(
            [
                ("r", "i1", 0.37),
                ("i1", "l1", 1.1),
                ("i1", "l2", 2.2),
            ]
        )
        msa = {
            'l1': 'AGFYLTV',
            'l2': 'DPHISKQ',
        }
        contact_map = np.array(
            [
                [1, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 1, 0],
                [0, 0, 1, 1, 0, 0, 0],
                [0, 0, 1, 1, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0],
                [0, 1, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 1],
            ]
        )
        # contact_map = np.eye(7)
        site_rates = [1.0, 2.0, 2.0, 2.0, 1.0, 2.0, 1.0]  # I use 2s to make sure site rates are not getting used for coevolution
        wag = wag_matrix().to_numpy()
        pi = compute_stationary_distribution(wag)
        wag_x_wag = chain_product(wag, wag)
        np.testing.assert_almost_equal(
            matrix_exponential(wag_x_wag)[0, 0],
            matrix_exponential(wag)[0, 0] ** 2
        )
        pi_x_pi = compute_stationary_distribution(wag_x_wag)
        ll, lls = likelihood_computation_wrapper(
            tree=tree,
            msa=msa,
            contact_map=contact_map,
            site_rates=site_rates,
            amino_acids=src.utils.amino_acids,
            pi_1=pi,
            Q_1=wag,
            pi_2=pi_x_pi,
            Q_2=wag_x_wag,
            method="python",
        )
        lls_expected = [
            -5.301370,
            (-5.790787 + -5.537568) / 2,
            (-6.895662 + -6.497235) / 2,
            (-6.497235 + -6.895662) / 2,
            -5.436122,
            (-5.537568 + -5.790787) / 2,
            -6.212303,
        ]
        np.testing.assert_almost_equal(lls, lls_expected, decimal=4)
        np.testing.assert_almost_equal(ll, sum(lls_expected), decimal=4)

    def test_small_wag_x_wag_2_seqs_many_sites_and_gaps(self):
        """
        This was manually verified with FastTree.
        """
        tree = Tree()
        tree.add_nodes(["r", "i1", "l1", "l2"])
        tree.add_edges(
            [
                ("r", "i1", 0.37),
                ("i1", "l1", 1.1),
                ("i1", "l2", 2.2),
            ]
        )
        msa = {
            'l1': '----LT-',
            'l2': '-PHI--Q',
        }
        contact_map = np.array(
            [
                [1, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 1, 0],
                [0, 0, 1, 1, 0, 0, 0],
                [0, 0, 1, 1, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0],
                [0, 1, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 1],
            ]
        )
        # contact_map = np.eye(7)
        site_rates = [1.0, 2.0, 2.0, 2.0, 1.0, 2.0, 1.0]  # I use 2s to make sure site rates are not getting used for coevolution
        wag = wag_matrix().to_numpy()
        pi = compute_stationary_distribution(wag)
        wag_x_wag = chain_product(wag, wag)
        np.testing.assert_almost_equal(
            matrix_exponential(wag_x_wag)[0, 0],
            matrix_exponential(wag)[0, 0] ** 2
        )
        pi_x_pi = compute_stationary_distribution(wag_x_wag)
        ll, lls = likelihood_computation_wrapper(
            tree=tree,
            msa=msa,
            contact_map=contact_map,
            site_rates=site_rates,
            amino_acids=src.utils.amino_acids,
            pi_1=pi,
            Q_1=wag,
            pi_2=pi_x_pi,
            Q_2=wag_x_wag,
            method="python",
        )
        lls_expected = [
            0.0,
            (-3.084277 + -2.796673) / 2,
            (-3.711890 + -3.026892) / 2,
            (-3.026892 + -3.711890) / 2,
            -2.450980,
            (-2.796673 + -3.084277) / 2,
            -3.304213,
        ]
        np.testing.assert_almost_equal(lls, lls_expected, decimal=4)
        np.testing.assert_almost_equal(ll, sum(lls_expected), decimal=4)

    @parameterized.expand(
        [("1_cat", 1, -4649.6146), ("2_cat", 2, -4397.8184), ("4_cat", 4, -4337.8688), ("20_cat", 20, -4307.0638)]
    )
    @pytest.mark.slow
    def test_real_data_single_site(self, name, num_cats, ll_expected):
        """
        Test on family 1a92_1_A using only WAG (no co-evolution model).
        """
        tree = read_tree(os.path.join(DATA_DIR, f"tree_dir_{num_cats}_cat_wag/1a92_1_A.txt"))
        msa = read_msa(os.path.join(DATA_DIR, "msa_dir/1a92_1_A.txt"))
        site_rates = read_site_rates(os.path.join(DATA_DIR, f"site_rates_dir_{num_cats}_cat_wag/1a92_1_A.txt"))
        # contact_map = read_contact_map(os.path.join(DATA_DIR, "contact_map_dir/1a92_1_A.txt"))
        contact_map = np.eye(len(site_rates))

        wag = wag_matrix().to_numpy()
        pi = compute_stationary_distribution(wag)
        wag_x_wag = chain_product(wag, wag)
        np.testing.assert_almost_equal(
            matrix_exponential(wag_x_wag)[0, 0],
            matrix_exponential(wag)[0, 0] ** 2
        )
        pi_x_pi = compute_stationary_distribution(wag_x_wag)
        ll, lls = likelihood_computation_wrapper(
            tree=tree,
            msa=msa,
            contact_map=contact_map,
            site_rates=site_rates,
            amino_acids=src.utils.amino_acids,
            pi_1=pi,
            Q_1=wag,
            pi_2=pi_x_pi,
            Q_2=wag_x_wag,
            method="python",
        )
        np.testing.assert_almost_equal(ll, ll_expected, decimal=4)

    @parameterized.expand(
        [("20_cat", 20, -264605.0691)]
    )
    @pytest.mark.slow
    def test_real_data_single_site_huge(self, name, num_cats, ll_expected):
        """
        Test on family 13gs_1_A using only WAG (no co-evolution model).
        """
        tree = read_tree(os.path.join(DATA_DIR, f"tree_dir_{num_cats}_cat_wag/13gs_1_A.txt"))
        msa = read_msa(os.path.join(DATA_DIR, "msa_dir/13gs_1_A.txt"))
        site_rates = read_site_rates(os.path.join(DATA_DIR, f"site_rates_dir_{num_cats}_cat_wag/13gs_1_A.txt"))
        # contact_map = read_contact_map(os.path.join(DATA_DIR, "contact_map_dir/13gs_1_A.txt"))
        contact_map = np.eye(len(site_rates))

        wag = wag_matrix().to_numpy()
        pi = compute_stationary_distribution(wag)
        wag_x_wag = chain_product(wag, wag)
        np.testing.assert_almost_equal(
            matrix_exponential(wag_x_wag)[0, 0],
            matrix_exponential(wag)[0, 0] ** 2
        )
        pi_x_pi = compute_stationary_distribution(wag_x_wag)
        ll, lls = likelihood_computation_wrapper(
            tree=tree,
            msa=msa,
            contact_map=contact_map,
            site_rates=site_rates,
            amino_acids=src.utils.amino_acids,
            pi_1=pi,
            Q_1=wag,
            pi_2=pi_x_pi,
            Q_2=wag_x_wag,
            method="python",
        )
        np.testing.assert_almost_equal(ll, ll_expected, decimal=2)

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

    #     ll, lls = likelihood_computation_wrapper(
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
    #         method="python",
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
