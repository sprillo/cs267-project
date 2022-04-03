import os
import tempfile
import unittest
from typing import Dict

import pandas as pd
from parameterized import parameterized

from src.simulation import simulate_msas
from src.io import read_msa


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


class TestSimulationTiny(unittest.TestCase):
    @parameterized.expand(
        [("3 processes", 3), ("2 processes", 2), ("serial", 1)]
    )
    def test_simulate_msas_1(self, name, num_processes):
        with tempfile.TemporaryDirectory() as root_dir:
            root_dir = "test_output/"
            outdir = os.path.join(root_dir, "simulated_msas")
            families = ["fam1", "fam2", "fam3"]
            simulate_msas(
                tree_dir="./test_input_data/tiny/tree_dir",
                site_rates_dir="./test_input_data/tiny/site_rates_dir",
                contact_map_dir="./test_input_data/tiny/contact_map_dir_for_simulation",
                families=families,
                amino_acids=["I", "L", "S", "T"],
                pi_1_path="./test_input_data/tiny/test_simulate_msas_1/pi_1.txt",
                Q_1_path="./test_input_data/tiny/test_simulate_msas_1/Q_1.txt",
                pi_2_path="./test_input_data/tiny/test_simulate_msas_1/pi_2.txt",
                Q_2_path="./test_input_data/tiny/test_simulate_msas_1/Q_2.txt",
                strategy="all_transitions",
                output_msa_dir=outdir,
                random_seed=0,
                num_processes=num_processes,
            )
            for family in families:
                expected_msa = read_msa(
                    f"test_input_data/tiny/test_simulate_msas_1/expected_msas/{family}.txt"
                )
                simulated_msa = read_msa(
                    os.path.join(outdir, "result.txt")
                )
                check_msas_are_equal(
                    expected_msa,
                    simulated_msa,
                )
