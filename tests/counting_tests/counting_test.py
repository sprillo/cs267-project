import os
import tempfile
import unittest
from filecmp import dircmp
from typing import Dict

import pandas as pd
from parameterized import parameterized

from src.counting.count_co_transitions import count_co_transitions
from src.counting.count_transitions import count_transitions
from src.io.count_matrices import read_count_matrices


def check_count_matrices_are_equal(
    count_matrices_1: Dict[str, pd.DataFrame],
    count_matrices_2: Dict[str, pd.DataFrame],
) -> None:
    qs_1 = sorted(list(count_matrices_1.keys()))
    qs_2 = sorted(list(count_matrices_2.keys()))
    if qs_1 != qs_2:
        raise Exception(
            f"Quantization values are different:\nExpected: {qs_1}\nvs\nObtained: {qs_2}"
        )
    for q in qs_1:
        count_matrix_1 = count_matrices_1[q]
        count_matrix_2 = count_matrices_2[q]
        if list(count_matrix_1.columns) != list(count_matrix_2.columns):
            raise Exception(
                f"Count matrix columns differ:\nExpected:\n{count_matrix_1.columns}\nvs\nObtained:\n{count_matrix_2.columns}"
            )
        if list(count_matrix_1.index) != list(count_matrix_2.index):
            raise Exception(
                f"Count matrix indices differ:\nExpected:\n{count_matrix_1.index}\nvs\nObtained:\n{count_matrix_2.index}"
            )
        if not (count_matrix_1 == count_matrix_2).all().all():
            raise Exception(
                f"Count matrix contents differ:\nExpected:\n{count_matrix_1}\nvs\nObtained:\n{count_matrix_2}"
            )
    return True


class TestCountTransitionsTiny(unittest.TestCase):
    @parameterized.expand([("multiprocess", 3), ("serial", 1)])
    def test_count_transitions_edges(self, name, num_processes):
        with tempfile.TemporaryDirectory() as root_dir:
            # root_dir = "test_output/"
            outdir = os.path.join(root_dir, "count_matrices_dir_edges")
            count_transitions(
                tree_dir="./test_input_data/tiny/tree_dir",
                msa_dir="./test_input_data/tiny/msa_dir",
                site_rates_dir="./test_input_data/tiny/site_rates_dir",
                families=["fam1", "fam2", "fam3"],
                # families=["fam3"],
                amino_acids=["I", "L", "S", "T"],
                quantization_points=[1.99, 5.01],
                edge_or_cherry="edge",
                output_count_matrices_dir=outdir,
                num_processes=num_processes,
            )
            count_matrices = read_count_matrices(
                os.path.join(outdir, "result.txt")
            )
            expected_count_matrices = read_count_matrices(
                "test_input_data/tiny/count_matrices_dir_edges/result.txt"
            )
            check_count_matrices_are_equal(
                expected_count_matrices,
                count_matrices,
            )

    @parameterized.expand([("multiprocess", 3), ("serial", 1)])
    def test_count_transitions_cherries(self, name, num_processes):
        with tempfile.TemporaryDirectory() as root_dir:
            # root_dir = "test_output/"
            outdir = os.path.join(root_dir, "count_matrices_dir_cherries")
            count_transitions(
                tree_dir="./test_input_data/tiny/tree_dir",
                msa_dir="./test_input_data/tiny/msa_dir",
                site_rates_dir="./test_input_data/tiny/site_rates_dir",
                families=["fam1", "fam2", "fam3"],
                # families=["fam3"],
                amino_acids=["I", "L", "S", "T"],
                quantization_points=[1.99, 10.01],
                edge_or_cherry="cherry",
                output_count_matrices_dir=outdir,
                num_processes=num_processes,
            )
            count_matrices = read_count_matrices(
                os.path.join(outdir, "result.txt")
            )
            expected_count_matrices = read_count_matrices(
                "test_input_data/tiny/count_matrices_dir_cherries/result.txt"
            )
            check_count_matrices_are_equal(
                expected_count_matrices,
                count_matrices,
            )

    @parameterized.expand([("multiprocess", 3), ("serial", 1)])
    def test_count_co_transitions_edges(self, name, num_processes):
        with tempfile.TemporaryDirectory() as root_dir:
            # root_dir = "test_output/"
            outdir = os.path.join(root_dir, "count_co_matrices_dir_edges")
            count_co_transitions(
                tree_dir="./test_input_data/tiny/tree_dir",
                msa_dir="./test_input_data/tiny/msa_dir",
                contact_map_dir="./test_input_data/tiny/contact_map_dir",
                families=["fam1", "fam2", "fam3"],
                # families=["fam3"],
                amino_acids=["I", "L", "S", "T"],
                quantization_points=[1.99, 5.01],
                edge_or_cherry="edge",
                minimum_distance_for_nontrivial_contact=2,
                output_count_matrices_dir=outdir,
                num_processes=num_processes,
            )
            count_matrices = read_count_matrices(
                os.path.join(outdir, "result.txt")
            )
            expected_count_matrices = read_count_matrices(
                "test_input_data/tiny/count_co_matrices_dir_edges/result.txt"
            )
            check_count_matrices_are_equal(
                expected_count_matrices,
                count_matrices,
            )

    @parameterized.expand([("multiprocess", 3), ("serial", 1)])
    def test_count_co_transitions_cherries(self, name, num_processes):
        with tempfile.TemporaryDirectory() as root_dir:
            # root_dir = "test_output/"
            outdir = os.path.join(root_dir, "count_co_matrices_dir_cherries")
            count_co_transitions(
                tree_dir="./test_input_data/tiny/tree_dir",
                msa_dir="./test_input_data/tiny/msa_dir",
                contact_map_dir="./test_input_data/tiny/contact_map_dir",
                families=["fam1", "fam2", "fam3"],
                # families=["fam3"],
                amino_acids=["I", "L", "S", "T"],
                quantization_points=[1.99, 10.01],
                edge_or_cherry="cherry",
                minimum_distance_for_nontrivial_contact=2,
                output_count_matrices_dir=outdir,
                num_processes=num_processes,
            )
            count_matrices = read_count_matrices(
                os.path.join(outdir, "result.txt")
            )
            expected_count_matrices = read_count_matrices(
                "test_input_data/tiny/count_co_matrices_dir_cherries/result.txt"
            )
            check_count_matrices_are_equal(
                expected_count_matrices,
                count_matrices,
            )
