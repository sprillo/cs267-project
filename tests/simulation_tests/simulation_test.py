import os
import tempfile
import unittest
from typing import Dict
from collections import defaultdict

import numpy as np
import pandas as pd
from parameterized import parameterized

from src.simulation import simulate_msas
from src.io import read_msa, write_contact_map, write_site_rates, read_tree


def create_synthetic_contact_map(
    num_sites: int,
    num_sites_in_contact: int,
    random_seed: int
) -> np.array:
    if num_sites_in_contact % 2 != 0:
        raise Exception(f"num_sites_in_contact should be even, but provided: {num_sites_in_contact}")
    num_contacting_pairs = num_sites_in_contact // 2
    contact_map = np.zeros(shape=(num_sites, num_sites), dtype=int)
    np.random.seed(random_seed)
    sites_in_contact = np.random.choice(range(num_sites), num_sites_in_contact, replace=False)
    contacting_pairs = [
        (sites_in_contact[2 * i], sites_in_contact[2 * i + 1])
        for i in range(num_contacting_pairs)
    ]
    for (i, j) in contacting_pairs:
        contact_map[i, j] = contact_map[j, i] = 1
    for i in range(num_sites):
        contact_map[i, i] = 1
    return contact_map


def check_msas_are_equal(
    msa_1: Dict[str, pd.DataFrame],
    msa_2: Dict[str, pd.DataFrame],
) -> None:
    seq_names_1 = sorted(list(msa_1.keys()))
    seq_names_2 = sorted(list(msa_2.keys()))
    if seq_names_1 != seq_names_2:
        raise Exception(
            f"Sequence names are different:\nExpected: "
            f"{seq_names_1}\nvs\nObtained: {seq_names_2}"
        )
    for seq_names in seq_names_1:
        seq_1 = msa_1[seq_names_1]
        seq_2 = msa_2[seq_names_2]
        if seq_1 != seq_2:
            raise Exception(
                f"Sequences differ:\nExpected:\n"
                f"{seq_1}\nvs\nObtained:\n"
                f"{seq_2}"
            )
    return True


class TestSimulation(unittest.TestCase):
    @parameterized.expand(
        [("3 processes", 3), ("2 processes", 2), ("serial", 1)]
    )
    def test_simulate_msas_1(self, name, num_processes):
        families = ["fam1", "fam2", "fam3"]
        tree_dir = "./tests/simulation_tests/test_input_data/tiny/tree_dir"
        with tempfile.TemporaryDirectory() as synthetic_contact_map_dir:
            synthetic_contact_map_dir = "./synthetic_contact_maps"
            # Create synthetic contact maps
            contact_maps = {}
            for i, family in enumerate(families):
                num_sites = 1000
                num_sites_in_contact = 600
                contact_map = create_synthetic_contact_map(
                    num_sites=num_sites,
                    num_sites_in_contact=num_sites_in_contact,
                    random_seed=i
                )
                contact_map_path = os.path.join(
                    synthetic_contact_map_dir,
                    family + ".txt"
                )
                write_contact_map(
                    contact_map,
                    contact_map_path
                )
                contact_maps[family] = contact_map
            with tempfile.TemporaryDirectory() as synthetic_site_rates_dir:
                synthetic_site_rates_dir = "./synthetic_site_rates"
                for i, family in enumerate(families):
                    site_rates = [1.0 * np.log(1 + i) for i in range(num_sites)]
                    site_rates_path = os.path.join(
                        synthetic_site_rates_dir,
                        family + ".txt"
                    )
                    write_site_rates(
                        site_rates,
                        site_rates_path
                    )
                with tempfile.TemporaryDirectory() as root_dir:
                    root_dir = "test_output/"
                    simulated_msa_dir = os.path.join(root_dir, "simulated_msas")
                    simulate_msas(
                        tree_dir=tree_dir,
                        site_rates_dir=synthetic_site_rates_dir,
                        contact_map_dir=synthetic_contact_map_dir,
                        families=families,
                        amino_acids=["S", "T"],
                        pi_1_path="./tests/simulation_tests/test_input_data/tiny/model/pi_1.txt",
                        Q_1_path="./tests/simulation_tests/test_input_data/tiny/model/Q_1.txt",
                        pi_2_path="./tests/simulation_tests/test_input_data/tiny/model/pi_2.txt",
                        Q_2_path="./tests/simulation_tests/test_input_data/tiny/model/Q_2.txt",
                        strategy="all_transitions",
                        output_msa_dir=simulated_msa_dir,
                        random_seed=0,
                        num_processes=num_processes,
                    )
                    # Check that the distribution of the endings states matches the stationary distribution
                    C_1 = defaultdict(int)  # single states
                    C_2 = defaultdict(int)  # co-evolving pairs
                    for family in families:
                        tree_path = os.path.join(
                            tree_dir,
                            family + ".txt"
                        )
                        tree = read_tree(
                            tree_path=tree_path,
                        )
                        msa = read_msa(
                            os.path.join(simulated_msa_dir, family + ".txt")
                        )
                        contacting_pairs = list(zip(*np.where(contact_map == 1)))
                        contacting_pairs = [
                            (i, j)
                            for (i, j) in contacting_pairs
                            if i < j
                        ]
                        contacting_sites = list(sum(contacting_pairs, ()))
                        sites_indep = [i for i in range(num_sites) if i not in contacting_sites]
                        for node in tree.nodes():
                            if node not in msa:
                                raise Exception(f"Missing sequence for node: {node}")
                            if tree.is_leaf(node):
                                seq = msa[node]
                                for i in sites_indep:
                                    state = seq[i]
                                    C_1[state] += 1
                                for (i, j) in contacting_pairs:
                                    state = seq[i] + seq[j]
                                    C_2[state] += 1
                    print(C_1)
                    print(C_2)
                    assert(False)

                    # for family in families:
                    #     expected_msa = read_msa(
                    #         f"./tests/simulation_tests/test_input_data/tiny/simulated_msas/{family}.txt"
                    #     )
                    #     simulated_msa = read_msa(
                    #         os.path.join(outdir, "result.txt")
                    #     )
                    #     check_msas_are_equal(
                    #         expected_msa,
                    #         simulated_msa,
                    #     )
