import os
import tempfile
import unittest
import filecmp

from ete3 import Tree as TreeETE
import json
from parameterized import parameterized

from src.phylogeny_estimation import fast_tree

from src.estimation import em_lg
from src.estimation._em_lg import _install_historian, _translate_tree_and_msa_to_stock_format, _translate_rate_matrix_to_historian_format, _translate_rate_matrix_from_historian_format
from src.markov_chain import get_lg_path
from src.utils import get_amino_acids

DATA_DIR = "./tests/estimation_tests/test_input_data"


class TestFastTree(unittest.TestCase):
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

    def test_translate_rate_matrix_to_historian_format(self):
        """
        Expected output is at ./test_input_data/historian_init.json
        """
        with tempfile.NamedTemporaryFile("w") as historian_init_file:
            historian_init_path = historian_init_file.name
            _translate_rate_matrix_to_historian_format(
                initialization_rate_matrix_path=get_lg_path(),
                historian_init_path=historian_init_path,
            )
            filepath_1 = f"{DATA_DIR}/historian_init.json"
            filepath_2 = historian_init_path
            assert(filecmp.cmp(filepath_1, filepath_2))

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

    # def test_translate_rate_matrix_from_historian_format(self):
    #     raise NotImplementedError
