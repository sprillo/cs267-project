import os
import tempfile
import unittest
import filecmp

import numpy as np
from ete3 import Tree as TreeETE
import json
from parameterized import parameterized

from collections import defaultdict

from src.estimation import em_lg
from src.estimation._em_lg import _install_historian, _translate_tree_and_msa_to_stock_format, _translate_rate_matrix_to_historian_format, _translate_rate_matrix_from_historian_format, _translate_rate_matrix_from_historian_format
from src.markov_chain import get_lg_path, get_lg_stationary_path, get_lg_x_lg_path, get_lg_x_lg_stationary_path, get_equ_path
from src.utils import get_amino_acids
from src.io import read_rate_matrix, write_contact_map, write_site_rates, read_tree, read_msa, read_probability_distribution
from src.simulation import simulate_msas
from tests.utils import create_synthetic_contact_map
from tests.simulation_tests.simulation_test import check_empirical_counts

DATA_DIR = "./tests/estimation_tests/test_input_data"


class TestEMLG(unittest.TestCase):
    def test_installation(self):
        """
        Test that Historian is installed
        """
        _install_historian()

    def test_translate_tree_and_msa_to_stock_format(self):
        """
        The expected output is at ./test_input_data/stock_dir/fam1_{i}.txt
        """
        with tempfile.TemporaryDirectory() as stock_dir:
            res = _translate_tree_and_msa_to_stock_format(
                "fam1",
                f"{DATA_DIR}/tree_dir",
                f"{DATA_DIR}/msa_dir",
                f"{DATA_DIR}/site_rates_dir",
                get_amino_acids(),
                stock_dir,
            )
            self.assertEqual(
                res,
                [f"fam1_{i}" for i in range(3)],
            )
            for i in range(3):
                filepath_1 = f"{DATA_DIR}/stock_dir/fam1_{i}.txt"
                filepath_2 = f"{stock_dir}/fam1_{i}.txt"
                assert(filecmp.cmp(filepath_1, filepath_2))

    def test_translate_tree_and_msa_to_stock_format_with_trifurcations(self):
        """
        The expected output is at ./test_input_data/stock_dir_trifurcation/fam1_{i}.txt
        """
        with tempfile.TemporaryDirectory() as stock_dir:
            stock_dir = "stock_dir"
            res = _translate_tree_and_msa_to_stock_format(
                "fam1",
                f"{DATA_DIR}/tree_dir_trifurcation",
                f"{DATA_DIR}/msa_dir",
                f"{DATA_DIR}/site_rates_dir",
                get_amino_acids(),
                stock_dir,
            )
            self.assertEqual(
                res,
                [f"fam1_{i}" for i in range(3)],
            )
            for i in range(3):
                filepath_1 = f"{DATA_DIR}/stock_dir_trifurcation/fam1_{i}.txt"
                filepath_2 = f"{stock_dir}/fam1_{i}.txt"
                assert(filecmp.cmp(filepath_1, filepath_2))

    def test_translate_rate_matrix_to_historian_format(self):
        """
        Expected output is at ./test_input_data/historian_init.json
        """
        with tempfile.NamedTemporaryFile("w") as historian_init_file:
            historian_init_path = historian_init_file.name
            historian_init_path = "historian_init_path"
            _translate_rate_matrix_to_historian_format(
                initialization_rate_matrix_path=get_lg_path(),
                historian_init_path=historian_init_path,
            )
            filepath_1 = f"{DATA_DIR}/historian_init.json"
            filepath_2 = historian_init_path
            file_1_lines = open(filepath_1).read().split('\n')
            file_2_lines = open(filepath_2).read().split('\n')
            for line_1, line_2 in zip(file_1_lines, file_2_lines):
                tokens_1, tokens_2 = line_1.split(), line_2.split()
                for token_1, token_2 in zip(tokens_1, tokens_2):
                    try:
                        np.testing.assert_almost_equal(float(token_1.strip(',')), float(token_2.strip(',')))
                    except Exception:
                        self.assertEqual(token_1, token_2)

    def test_run_historian_from_CLI(self):
        """
        Run Historian from CLI
        """
        with tempfile.NamedTemporaryFile("w") as historian_learned_rate_matrix_file:
            historian_learned_rate_matrix_path = historian_learned_rate_matrix_file.name
            command = (
                "src/estimation/historian/bin/historian fit"
                + ''.join([f" {DATA_DIR}/stock_dir/fam1_{i}.txt" for i in range(3)])
                + f" -model {DATA_DIR}/historian_init_small.json"
                + " -band 0"
                + f" -fixgaprates > {historian_learned_rate_matrix_path} -v2"
            )
            print(f"Going to run: {command}")
            os.system(command)
            with open(historian_learned_rate_matrix_path) as json_file:
                learned_rate_matrix_json = json.load(json_file)
                assert("subrate" in learned_rate_matrix_json.keys())

    def test_translate_rate_matrix_from_historian_format(self):
        with tempfile.NamedTemporaryFile("w") as learned_rate_matrix_file:
            learned_rate_matrix_path = learned_rate_matrix_file.name
            _translate_rate_matrix_from_historian_format(
                f"{DATA_DIR}/historian_learned_rate_matrix.json",
                alphabet=["A", "R", "N", "Q"],
                learned_rate_matrix_path=learned_rate_matrix_path,
            )
            filepath_1 = f"{DATA_DIR}/learned_rate_matrix.txt"
            filepath_2 = learned_rate_matrix_path
            assert(filecmp.cmp(filepath_1, filepath_2))

    def test_run_historian_from_python_api(self):
        """
        Run Historian from our CLI
        """
        with tempfile.TemporaryDirectory() as learned_rate_matrix_dir:
            em_lg(
                tree_dir=f"{DATA_DIR}/tree_dir",
                msa_dir=f"{DATA_DIR}/msa_dir",
                site_rates_dir=f"{DATA_DIR}/site_rates_dir",
                families=["fam1"],
                initialization_rate_matrix_path=f"{DATA_DIR}/historian_init_small.txt",
                output_rate_matrix_dir=learned_rate_matrix_dir,
                extra_command_line_args="-band 0 -fixgaprates",
            )
            learned_rate_matrix = read_rate_matrix(
                os.path.join(learned_rate_matrix_dir, "result.txt")
            )
            np.testing.assert_almost_equal(
                learned_rate_matrix.to_numpy(),
                read_rate_matrix(f"{DATA_DIR}/learned_rate_matrix.txt").to_numpy(),
            )

    # def test_run_historian_to_recover_rate_matrix(self):
    #     """
    #     We run historian on simulated data from 2 state model
    #     """
    #     DATA_DIR_SIMULATION = "./tests/simulation_tests/test_input_data"
    #     with tempfile.TemporaryDirectory() as learned_rate_matrix_dir:
    #         learned_rate_matrix_dir = "learned_rate_matrix_dir"
    #         families = ["fam1", "fam2", "fam3"]
    #         tree_dir = f"{DATA_DIR}/tree_dir"
    #         with tempfile.TemporaryDirectory() as synthetic_contact_map_dir:
    #             # synthetic_contact_map_dir = "tests/simulation_tests/test_input_data/synthetic_contact_map_dir"  # noqa
    #             contact_maps = {}
    #             for i, family in enumerate(families):
    #                 num_sites = 1000
    #                 num_sites_in_contact = 0
    #                 contact_map = create_synthetic_contact_map(
    #                     num_sites=num_sites,
    #                     num_sites_in_contact=num_sites_in_contact,
    #                     random_seed=i,
    #                 )
    #                 contact_map_path = os.path.join(
    #                     synthetic_contact_map_dir, family + ".txt"
    #                 )
    #                 write_contact_map(contact_map, contact_map_path)
    #                 contact_maps[family] = contact_map

    #             with tempfile.TemporaryDirectory() as synthetic_site_rates_dir:
    #                 # synthetic_site_rates_dir = "tests/simulation_tests/test_input_data/synthetic_site_rates_dir"  # noqa
    #                 for i, family in enumerate(families):
    #                     site_rates = [0.25, 0.5, 1.0, 1.5, 2.0] * 200
    #                     site_rates_path = os.path.join(
    #                         synthetic_site_rates_dir, family + ".txt"
    #                     )
    #                     write_site_rates(site_rates, site_rates_path)

    #                 with tempfile.TemporaryDirectory() as simulated_msa_dir:
    #                     # simulated_msa_dir = "tests/simulation_tests/test_input_data/simulated_msa_dir"  # noqa
    #                     simulated_msa_dir = "simulated_msa_dir"
    #                     simulate_msas(
    #                         tree_dir=tree_dir,
    #                         site_rates_dir=synthetic_site_rates_dir,
    #                         contact_map_dir=synthetic_contact_map_dir,
    #                         families=families,
    #                         amino_acids=["S", "T"],
    #                         pi_1_path=f"{DATA_DIR}/pi_1.txt",
    #                         Q_1_path=f"{DATA_DIR}/Q_1.txt",
    #                         pi_2_path=f"{DATA_DIR_SIMULATION}/normal_model/pi_2.txt",
    #                         Q_2_path=f"{DATA_DIR_SIMULATION}/normal_model/Q_2.txt",
    #                         strategy="all_transitions",
    #                         output_msa_dir=simulated_msa_dir,
    #                         random_seed=0,
    #                         num_processes=3,
    #                     )
    #                     # Check that the distribution of the endings states matches
    #                     # the stationary distribution
    #                     C_1 = defaultdict(int)  # single states
    #                     for family in families:
    #                         tree_path = os.path.join(tree_dir, family + ".txt")
    #                         tree = read_tree(
    #                             tree_path=tree_path,
    #                         )
    #                         msa = read_msa(
    #                             os.path.join(simulated_msa_dir, family + ".txt")
    #                         )
    #                         sites_indep = [
    #                             i
    #                             for i in range(num_sites)
    #                         ]
    #                         for node in tree.nodes():
    #                             if node not in msa:
    #                                 raise Exception(
    #                                     f"Missing sequence for node: {node}"
    #                                 )
    #                             if tree.is_leaf(node):
    #                                 seq = msa[node]
    #                                 for i in sites_indep:
    #                                     state = seq[i]
    #                                     C_1[state] += 1

    #                     pi_1 = read_probability_distribution(
    #                         f"{DATA_DIR}/pi_1.txt"
    #                     )
    #                     check_empirical_counts(C_1, pi_1, rel_error_tolerance=0.10)

    #                     em_lg(
    #                         tree_dir=tree_dir,
    #                         msa_dir=simulated_msa_dir,
    #                         site_rates_dir=synthetic_site_rates_dir,
    #                         families=["fam1", "fam2", "fam3"],
    #                         initialization_rate_matrix_path=f"{DATA_DIR}/historian_init_small_2_states.txt",
    #                         output_rate_matrix_dir=learned_rate_matrix_dir,
    #                     )
    #                     learned_rate_matrix = read_rate_matrix(
    #                         os.path.join(learned_rate_matrix_dir, "result.txt")
    #                     )
    #                     np.testing.assert_almost_equal(
    #                         learned_rate_matrix.to_numpy(),
    #                         read_rate_matrix(f"{DATA_DIR_SIMULATION}/normal_model/Q_1.txt").to_numpy(),
    #                     )

    # def test_run_historian_to_recover_rate_matrix_lg(self):
    #     """
    #     We run historian on simulated data from LG model
    #     """
    #     DATA_DIR_SIMULATION = "./tests/simulation_tests/test_input_data"
    #     with tempfile.TemporaryDirectory() as learned_rate_matrix_dir:
    #         learned_rate_matrix_dir = "learned_rate_matrix_dir"
    #         families = ["fam1", "fam2", "fam3"]
    #         tree_dir = f"{DATA_DIR}/tree_dir"
    #         with tempfile.TemporaryDirectory() as synthetic_contact_map_dir:
    #             # synthetic_contact_map_dir = "tests/simulation_tests/test_input_data/synthetic_contact_map_dir"  # noqa
    #             contact_maps = {}
    #             for i, family in enumerate(families):
    #                 num_sites = 1000
    #                 num_sites_in_contact = 0
    #                 contact_map = create_synthetic_contact_map(
    #                     num_sites=num_sites,
    #                     num_sites_in_contact=num_sites_in_contact,
    #                     random_seed=i,
    #                 )
    #                 contact_map_path = os.path.join(
    #                     synthetic_contact_map_dir, family + ".txt"
    #                 )
    #                 write_contact_map(contact_map, contact_map_path)
    #                 contact_maps[family] = contact_map

    #             with tempfile.TemporaryDirectory() as synthetic_site_rates_dir:
    #                 # synthetic_site_rates_dir = "tests/simulation_tests/test_input_data/synthetic_site_rates_dir"  # noqa
    #                 for i, family in enumerate(families):
    #                     site_rates = [0.25, 0.5, 1.0, 1.5, 2.0] * 200
    #                     site_rates_path = os.path.join(
    #                         synthetic_site_rates_dir, family + ".txt"
    #                     )
    #                     write_site_rates(site_rates, site_rates_path)

    #                 with tempfile.TemporaryDirectory() as simulated_msa_dir:
    #                     # simulated_msa_dir = "tests/simulation_tests/test_input_data/simulated_msa_dir"  # noqa
    #                     simulated_msa_dir = "simulated_msa_dir"
    #                     simulate_msas(
    #                         tree_dir=tree_dir,
    #                         site_rates_dir=synthetic_site_rates_dir,
    #                         contact_map_dir=synthetic_contact_map_dir,
    #                         families=families,
    #                         amino_acids=get_amino_acids(),
    #                         pi_1_path=get_lg_stationary_path(),
    #                         Q_1_path=get_lg_path(),
    #                         pi_2_path=get_lg_x_lg_stationary_path(),
    #                         Q_2_path=get_lg_x_lg_path(),
    #                         strategy="all_transitions",
    #                         output_msa_dir=simulated_msa_dir,
    #                         random_seed=0,
    #                         num_processes=3,
    #                     )

    #                     em_lg(
    #                         tree_dir=tree_dir,
    #                         msa_dir=simulated_msa_dir,
    #                         site_rates_dir=synthetic_site_rates_dir,
    #                         families=["fam1", "fam2", "fam3"],
    #                         initialization_rate_matrix_path=get_equ_path(),
    #                         output_rate_matrix_dir=learned_rate_matrix_dir,
    #                     )
    #                     learned_rate_matrix = read_rate_matrix(
    #                         os.path.join(learned_rate_matrix_dir, "result.txt")
    #                     )
    #                     np.testing.assert_almost_equal(
    #                         learned_rate_matrix.to_numpy(),
    #                         read_rate_matrix(get_lg_path()).to_numpy(),
    #                     )
