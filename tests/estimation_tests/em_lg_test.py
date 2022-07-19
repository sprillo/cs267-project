import os
import tempfile
import unittest
import filecmp

from ete3 import Tree as TreeETE
from parameterized import parameterized

from src.phylogeny_estimation import fast_tree

from src.estimation import em_lg
from src.estimation._em_lg import _install_historian, _translate_tree_and_msa_to_stock_format, _translate_rate_matrix_to_historian_format, _translate_rate_matrix_from_historian_format

DATA_DIR = "./tests/estimation_tests/test_input_data"


class TestFastTree(unittest.TestCase):
    def test_installation(self):
        """
        Test that Historian is installed
        """
        _install_historian()

    def test_translate_tree_and_msa_to_stock_format(self):
        """
        The expected output is at stock_dir/fam1_{i}.txt
        """
        with tempfile.TemporaryDirectory() as stock_dir:
            _translate_tree_and_msa_to_stock_format(
                "fam1",
                f"{DATA_DIR}/tree_dir",
                f"{DATA_DIR}/msa_dir",
                f"{DATA_DIR}/site_rates_dir",
                stock_dir,
            )
            for i in range(3):
                filepath_1 = f"{DATA_DIR}/stock_dir/fam1_{i}.txt"
                filepath_2 = f"{stock_dir}/fam1_{i}.txt"
                assert(filecmp.cmp(filepath_1, filepath_2))

    # def test_translate_rate_matrix_to_historian_format(self):
    #     raise NotImplementedError

    # def test_translate_rate_matrix_from_historian_format(self):
    #     raise NotImplementedError
