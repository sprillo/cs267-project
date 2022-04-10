import os
import tempfile
import unittest
from typing import Dict, List, Tuple

import itertools
import numpy as np
import pandas as pd
import pytest
import random
from parameterized import parameterized

from src.counting import count_co_transitions
from src.counting import count_transitions
from src.io import read_count_matrices
from src.evaluation import compute_log_likelihoods
from tests.utils import create_synthetic_contact_map


from src.io import Tree


def create_fake_msa_and_contact_map_and_site_rates(
    tree: Tree,
    amino_acids: List[str],
    random_seed: int,
    num_rate_categories: int,
) -> Tuple[Dict[str, str], np.array, List[float]]:
    """
    Create fake data from a tree.
    """
    np.random.seed(random_seed)
    random.seed(random_seed)

    num_leaves = sum([tree.is_leaf(v) for v in tree.nodes()])
    single_site_patterns = [''.join(pattern) for pattern in list(itertools.product(amino_acids, repeat=num_leaves))]
    pair_of_site_patterns = list(itertools.product(single_site_patterns, repeat=2))
    # Combine single-site paaterns and pair-of-site patterns
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
    # Validate that each site is in contact with at most one other site
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
        f"seq{i + 1}": ''.join(msa_array[:, i]) for i in range(num_leaves)
    }
    # print(f"msa = {msa}")
    site_rates = [0.5 * np.log(2 + i) for i in range(num_rate_categories)] * (int(num_sites / num_rate_categories) + 1)
    site_rates = site_rates[:num_sites]
    np.random.shuffle(site_rates)
    # print(f"site_rates = {site_rates}")
    return msa, contact_map, site_rates


class TestComputeLogLikelihoods(unittest.TestCase):
    # @parameterized.expand(
    #     [("3 processes", 3)]
    # )
    # def test_1(self, name, num_processes):
    #     tree = Tree()
    #     tree.add_nodes(["r", "i0", "l1", "l2"])
    #     tree.add_edges(
    #         [
    #             ("r", "i0", 0.14),
    #             ("i0", "l1", 1.14),
    #             ("i0", "l2", 0.71),
    #         ]
    #     )
    #     msa, contact_map, site_rates = \
    #         create_fake_msa_and_contact_map_and_site_rates(
    #             tree=tree,
    #             amino_acids=["G", "P"],
    #             random_seed=1
    #         )
    #     assert(False)

    @parameterized.expand(
        [("3 processes", 3)]
    )
    def test_2(self, name, num_processes):
        tree = Tree()
        tree.add_nodes(["r", "i0", "l1", "l2", "l3"])
        tree.add_edges(
            [
                ("r", "i0", 0.14),
                ("i0", "l1", 1.14),
                ("i0", "l2", 0.71),
                ("r", "l3", 3.14),
            ]
        )
        msa, contact_map, site_rates = \
            create_fake_msa_and_contact_map_and_site_rates(
                tree=tree,
                amino_acids=["G", "P"],
                random_seed=1,
                num_rate_categories=3
            )
        print(f"msa = {msa}")
        assert(False)