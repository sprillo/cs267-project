import os
import tempfile
import unittest

from src.estimation import quantized_transitions_mle


class TestQuantizedTransitionsMLE(unittest.TestCase):
    def test_smoke_toy_matrix(self):
        """
        Test that RateMatrixLearner runs on a very small input dataset.
        """
        with tempfile.TemporaryDirectory() as root_dir:
            outdir = os.path.join(root_dir, "Q1_estimate")
            initialization_path = (
                "tests/test_input_data"
                "/3x3_pande_reversible_initialization.txt"
            )
            quantized_transitions_mle(
                count_matrices_path="tests/test_input_data/matrices_toy.txt",
                initialization_path=initialization_path,
                mask_path=None,
                output_rate_matrix_dir=outdir,
                stationary_distribution_path=None,
                rate_matrix_parameterization="pande_reversible",
                device="cpu",
                learning_rate=1e-1,
                num_epochs=3,
                do_adam=True,
            )

    # def test_smoke_toy_matrix_raises_if_mask_and_initialization_incompatible(self):
    #     """
    #     Test that RateMatrixLearner raises error if mask and
    #     initialization are incompatible.
    #     """
    #     with tempfile.TemporaryDirectory() as root_dir:
    #         outdir = os.path.join(root_dir, 'Q1_estimate')
    #         for use_cached in [True]:
    #             with self.assertRaises(ValueError):
    #                 rate_matrix_learner = RateMatrixLearner(
    #                     frequency_matrices="test_input_data/matrices_toy.txt",
    #                     output_dir=outdir,
    #                     stationnary_distribution=None,
    #                     mask="test_input_data/3x3_mask.txt",
    #                     # frequency_matrices_sep=",",
    #                     rate_matrix_parameterization="pande_reversible",
    #                     device='cpu',
    #                     use_cached=use_cached,
    #                     initialization=np.loadtxt("test_input_data/3x3_pande_reversible_initialization.txt"),
    #                 )
    #                 rate_matrix_learner.train(
    #                     lr=1e-1,
    #                     num_epochs=3,
    #                     do_adam=True,
    #                 )

    # def test_smoke_toy_matrix_mask(self):
    #     """
    #     Test that RateMatrixLearner runs on a very small input dataset,
    #     using masking.
    #     """
    #     with tempfile.TemporaryDirectory() as root_dir:
    #         outdir = os.path.join(root_dir, 'Q1_estimate')
    #         for use_cached in [False, True]:
    #             rate_matrix_learner = RateMatrixLearner(
    #                 frequency_matrices="test_input_data/matrices_toy.txt",
    #                 output_dir=outdir,
    #                 stationnary_distribution=None,
    #                 mask="test_input_data/3x3_mask.txt",
    #                 # frequency_matrices_sep=",",
    #                 rate_matrix_parameterization="pande_reversible",
    #                 device='cpu',
    #                 use_cached=use_cached,
    #                 initialization=np.loadtxt("test_input_data/3x3_pande_reversible_initialization_mask.txt"),
    #             )
    #             rate_matrix_learner.train(
    #                 lr=1e-1,
    #                 num_epochs=3,
    #                 do_adam=True,
    #             )
    #             # Check that the learned rate matrix has the right masking
    #             # structure
    #             mask = np.loadtxt("test_input_data/3x3_mask.txt")
    #             learned_rate_matrix = np.loadtxt(os.path.join(outdir, "learned_matrix.txt"))
    #             np.testing.assert_almost_equal(mask == 1.0, learned_rate_matrix != 0.0)

    # def test_existing_results_are_not_overwritten(self):
    #     """
    #     We want to make sure we don't corrupt previous runs accidentaly.
    #     """
    #     with tempfile.TemporaryDirectory() as root_dir:
    #         outdir = os.path.join(root_dir, 'Q1_estimate')
    #         for i, use_cached in enumerate([False, False]):
    #             rate_matrix_learner = RateMatrixLearner(
    #                 frequency_matrices="test_input_data/matrices_toy.txt",
    #                 output_dir=outdir,
    #                 stationnary_distribution=None,
    #                 mask=None,
    #                 # frequency_matrices_sep=",",
    #                 rate_matrix_parameterization="pande_reversible",
    #                 device='cpu',
    #                 use_cached=use_cached,
    #             )
    #             if i == 0:
    #                 rate_matrix_learner.train(
    #                     lr=1e-1,
    #                     num_epochs=3,
    #                     do_adam=True,
    #                 )
    #             else:
    #                 with self.assertRaises(PermissionError):
    #                     rate_matrix_learner.train(
    #                         lr=1e-1,
    #                         num_epochs=3,
    #                         do_adam=True,
    #                     )

    # def test_smoke_large_matrix(self):
    #     """
    #     Test that RateMatrixLearner runs on a large input dataset.
    #     """
    #     with tempfile.TemporaryDirectory() as root_dir:
    #         outdir = os.path.join(root_dir, 'Q1_estimate')
    #         for use_cached in [False, True]:
    #             rate_matrix_learner = RateMatrixLearner(
    #                 frequency_matrices="test_input_data/matrices_small/matrices_by_quantized_branch_length.txt",
    #                 output_dir=outdir,
    #                 stationnary_distribution=None,
    #                 mask="test_input_data/20x20_random_mask.txt",
    #                 # frequency_matrices_sep=",",
    #                 rate_matrix_parameterization="pande_reversible",
    #                 device='cpu',
    #                 use_cached=use_cached,
    #             )
    #             rate_matrix_learner.train(
    #                 lr=1e-1,
    #                 num_epochs=3,
    #                 do_adam=True,
    #             )
    #             # Test that the masking worked.
    #             mask = np.loadtxt("test_input_data/20x20_random_mask.txt")
    #             learned_rate_matrix = np.loadtxt(os.path.join(outdir, "learned_matrix.txt"))
    #             np.testing.assert_almost_equal(mask == 1.0, learned_rate_matrix != 0.0)

    # def test_smoke_huge_matrix(self):
    #     """
    #     Test that RateMatrixLearner runs on a huge input dataset.
    #     """
    #     with tempfile.TemporaryDirectory() as root_dir:
    #         outdir = os.path.join(root_dir, 'Q2_estimate')
    #         for use_cached in [False, True]:
    #             rate_matrix_learner = RateMatrixLearner(
    #                 frequency_matrices="test_input_data/co_matrices_small/matrices_by_quantized_branch_length.txt",
    #                 output_dir=outdir,
    #                 stationnary_distribution=None,
    #                 mask="input_data/synthetic_rate_matrices/mask_Q2.txt",
    #                 # frequency_matrices_sep=",",
    #                 rate_matrix_parameterization="pande_reversible",
    #                 device='cpu',
    #                 use_cached=use_cached,
    #             )
    #             rate_matrix_learner.train(
    #                 lr=1e-1,
    #                 num_epochs=1,
    #                 do_adam=True,
    #             )
    #             # Test that the masking worked.
    #             mask = np.loadtxt("input_data/synthetic_rate_matrices/mask_Q2.txt")
    #             learned_rate_matrix = np.loadtxt(os.path.join(outdir, "learned_matrix.txt"))
    #             np.testing.assert_almost_equal(mask == 1.0, learned_rate_matrix != 0.0)
