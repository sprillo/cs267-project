import os
import tempfile
import unittest

import numpy as np

from src.estimation import quantized_transitions_mle
from src.io import read_mask_matrix, read_rate_matrix


class TestQuantizedTransitionsMLE(unittest.TestCase):
    def test_smoke_toy_matrix(self):
        """
        Test that RateMatrixLearner runs on a very small input dataset.
        """
        with tempfile.TemporaryDirectory() as output_rate_matrix_dir:
            initialization_path = (
                "tests/test_input_data"
                "/3x3_pande_reversible_initialization.txt"
            )
            quantized_transitions_mle(
                count_matrices_path="tests/test_input_data/matrices_toy.txt",
                initialization_path=initialization_path,
                mask_path=None,
                output_rate_matrix_dir=output_rate_matrix_dir,
                stationary_distribution_path=None,
                rate_matrix_parameterization="pande_reversible",
                device="cpu",
                learning_rate=1e-1,
                num_epochs=3,
                do_adam=True,
            )

    def test_smoke_toy_matrix_raises_if_mask_and_initialization_incompatible(
        self,
    ):
        """
        Test that RateMatrixLearner raises error if mask and
        initialization are incompatible.
        """
        with tempfile.TemporaryDirectory() as output_rate_matrix_dir:
            with self.assertRaises(ValueError):
                count_matrices_path = "tests/test_input_data/matrices_toy.txt"
                initialization_path = (
                    "tests/test_input_data"
                    "/3x3_pande_reversible_initialization.txt"
                )
                quantized_transitions_mle(
                    count_matrices_path=count_matrices_path,
                    initialization_path=initialization_path,
                    mask_path="tests/test_input_data/3x3_mask.txt",
                    output_rate_matrix_dir=output_rate_matrix_dir,
                    stationary_distribution_path=None,
                    rate_matrix_parameterization="pande_reversible",
                    device="cpu",
                    learning_rate=1e-1,
                    num_epochs=3,
                    do_adam=True,
                )

    def test_smoke_toy_matrix_mask(self):
        """
        Test that RateMatrixLearner runs on a very small input dataset,
        using masking.
        """
        with tempfile.TemporaryDirectory() as output_rate_matrix_dir:
            initialization_path = (
                "tests/test_input_data"
                "/3x3_pande_reversible_initialization_mask.txt"
            )
            quantized_transitions_mle(
                count_matrices_path="tests/test_input_data/matrices_toy.txt",
                initialization_path=initialization_path,
                mask_path="tests/test_input_data/3x3_mask.txt",
                output_rate_matrix_dir=output_rate_matrix_dir,
                stationary_distribution_path=None,
                rate_matrix_parameterization="pande_reversible",
                device="cpu",
                learning_rate=1e-1,
                num_epochs=3,
                do_adam=True,
            )
            # Check that the learned rate matrix has the right masking
            # structure
            mask = read_mask_matrix(
                "tests/test_input_data/3x3_mask.txt"
            ).to_numpy()
            learned_rate_matrix = read_rate_matrix(
                os.path.join(output_rate_matrix_dir, "result.txt")
            ).to_numpy()
            np.testing.assert_almost_equal(
                mask == 1, learned_rate_matrix != 0.0
            )

    def test_smoke_large_matrix(self):
        """
        Test that RateMatrixLearner runs on a large input dataset.
        """
        with tempfile.TemporaryDirectory() as output_rate_matrix_dir:
            count_matrices_path = (
                "tests/test_input_data/matrices_small"
                "/matrices_by_quantized_branch_length.txt"
            )
            quantized_transitions_mle(
                count_matrices_path=count_matrices_path,
                initialization_path=None,
                mask_path="tests/test_input_data/20x20_random_mask.txt",
                output_rate_matrix_dir=output_rate_matrix_dir,
                stationary_distribution_path=None,
                rate_matrix_parameterization="pande_reversible",
                device="cpu",
                learning_rate=1e-1,
                num_epochs=3,
                do_adam=True,
            )
            # Test that the masking worked.
            mask = read_mask_matrix(
                "tests/test_input_data/20x20_random_mask.txt"
            ).to_numpy()
            learned_rate_matrix = read_rate_matrix(
                os.path.join(output_rate_matrix_dir, "result.txt")
            ).to_numpy()
            np.testing.assert_almost_equal(
                mask == 1, learned_rate_matrix != 0.0
            )

    def test_smoke_huge_matrix(self):
        """
        Test that RateMatrixLearner runs on a huge input dataset.
        """
        with tempfile.TemporaryDirectory() as output_rate_matrix_dir:
            count_matrices_path = (
                "tests/test_input_data/co_matrices_small"
                "/matrices_by_quantized_branch_length.txt"
            )
            mask_path = (
                "tests/test_input_data/synthetic_rate_matrices" "/mask_Q2.txt"
            )
            quantized_transitions_mle(
                count_matrices_path=count_matrices_path,
                initialization_path=None,
                mask_path=mask_path,
                output_rate_matrix_dir=output_rate_matrix_dir,
                stationary_distribution_path=None,
                rate_matrix_parameterization="pande_reversible",
                device="cpu",
                learning_rate=1e-1,
                num_epochs=3,
                do_adam=True,
            )
            # Test that the masking worked.
            mask = read_mask_matrix(
                "tests/test_input_data/synthetic_rate_matrices/mask_Q2.txt"
            ).to_numpy()
            learned_rate_matrix = read_rate_matrix(
                os.path.join(output_rate_matrix_dir, "result.txt")
            ).to_numpy()
            np.testing.assert_almost_equal(
                mask == 1, learned_rate_matrix != 0.0
            )
