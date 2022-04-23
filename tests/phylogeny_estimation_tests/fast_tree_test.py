import os
import unittest
import tempfile
from filecmp import dircmp
from parameterized import parameterized

from src.phylogeny_estimation import fast_tree

from ete3 import Tree


def branch_length_l1_error(tree_true_path, tree_inferred_path) -> float:
    tree1 = Tree(tree_true_path)
    tree2 = Tree(tree_inferred_path)

    def dfs_branch_length_l1_error(v1, v2) -> float:
        l1_error = 0
        for (u1, u2) in zip(v1.children, v2.children):
            l1_error += abs(u1.dist - u2.dist)
            l1_error += dfs_branch_length_l1_error(u1, u2)
        return l1_error
    l1_error = dfs_branch_length_l1_error(tree1, tree2)
    return l1_error


class TestFastTree(unittest.TestCase):
    def test_basic_regression(self):
        """
        Test that FastTree runs and its output matches the expected output.
        The expected output is located at test_input_data/trees_small
        """
        with tempfile.TemporaryDirectory() as root_dir:
            outdir = os.path.join(root_dir, 'trees_small')
            families = ["1e7l_1_A", "5a0l_1_A", "6anz_1_B"]
            fast_tree(
                msa_dir='test_input_data/a3m_small',
                families=families,
                num_processes=n_process,
                outdir=outdir,
                rate_matrix='data/rate_matrices/wag.txt',
                num_rate_categories=20,
                output_tree_dir=root_dir,
            )
            for protein_family_name in families:
                tree_true_path = f"test_input_data/trees_small/{protein_family_name}.newick"
                tree_inferred_path = os.path.join(outdir, protein_family_name + '.newick')
                l1_error = branch_length_l1_error(tree_true_path, tree_inferred_path)
                assert(l1_error < 0.02)  # Redundant, but just in case

    # @parameterized.expand([("multiprocess", 3), ("single-process", 1)])
    # def test_custom_rate_matrix_runs_regression(self, name, n_process):
    #     """
    #     Tests the use of a custom rate matrix in FastTree.
    #     """
    #     with tempfile.TemporaryDirectory() as root_dir:
    #         outdir = os.path.join(root_dir, 'trees_small_Q1_uniform')
    #         phylogeny_generator = PhylogenyGenerator(
    #             a3m_dir_full='test_input_data/a3m_small',
    #             a3m_dir='test_input_data/a3m_small',
    #             n_process=n_process,
    #             expected_number_of_MSAs=3,
    #             outdir=outdir,
    #             max_seqs=8,
    #             max_sites=16,
    #             max_families=3,
    #             rate_matrix='input_data/synthetic_rate_matrices/Q1_uniform_FastTree.txt',
    #             fast_tree_cats=20,
    #             use_cached=False,
    #         )
    #         phylogeny_generator.run()
    #         dcmp = dircmp(outdir, 'test_input_data/trees_small_Q1_uniform')
    #         diff_files = dcmp.diff_files
    #         assert(len(diff_files) == 0)

    # @parameterized.expand([("multiprocess", 3), ("single-process", 1)])
    # def test_custom_rate_matrix_unnormalized_runs_regression(self, name, n_process):
    #     """
    #     Tests the use of an UNNORMALIZED custom rate matrix in FastTree.
    #     """
    #     with tempfile.TemporaryDirectory() as root_dir:
    #         outdir = os.path.join(root_dir, 'trees_small_Q1_uniform_halved')
    #         phylogeny_generator = PhylogenyGenerator(
    #             a3m_dir_full='test_input_data/a3m_small',
    #             a3m_dir='test_input_data/a3m_small',
    #             n_process=n_process,
    #             expected_number_of_MSAs=3,
    #             outdir=outdir,
    #             max_seqs=8,
    #             max_sites=16,
    #             max_families=3,
    #             rate_matrix='input_data/synthetic_rate_matrices/Q1_uniform_halved_FastTree.txt',
    #             fast_tree_cats=20,
    #             use_cached=False,
    #         )
    #         phylogeny_generator.run()
    #         for protein_family_name in ['1e7l_1_A', '5a0l_1_A', '6anz_1_B']:
    #             tree_true_path = f"test_input_data/trees_small_Q1_uniform_halved/{protein_family_name}.newick"
    #             tree_inferred_path = os.path.join(outdir, protein_family_name + '.newick')
    #             l1_error = branch_length_l1_error(tree_true_path, tree_inferred_path)
    #             assert(l1_error < 0.02)  # Redundant, but just in case

    # @parameterized.expand([("multiprocess", 3), ("single-process", 1)])
    # def test_inexistent_rate_matrix_raises_error(self, name, n_process):
    #     """
    #     If the rate matrix passed to FastTree does not exist, we should error out.
    #     """
    #     with tempfile.TemporaryDirectory() as root_dir:
    #         outdir = os.path.join(root_dir, 'trees')
    #         phylogeny_generator = PhylogenyGenerator(
    #             a3m_dir_full='test_input_data/a3m_small',
    #             a3m_dir='test_input_data/a3m_small',
    #             n_process=n_process,
    #             expected_number_of_MSAs=3,
    #             outdir=outdir,
    #             max_seqs=8,
    #             max_sites=16,
    #             max_families=3,
    #             rate_matrix='I-do-not-exist',
    #             fast_tree_cats=20,
    #             use_cached=False,
    #         )
    #         with self.assertRaises(PhylogenyGeneratorError):
    #             phylogeny_generator.run()

    # @parameterized.expand([("multiprocess", 3), ("single-process", 1)])
    # def test_malformed_a3m_file_raises_error(self, name, n_process):
    #     """
    #     If the a3m data is corrupted, an error should be raised.
    #     """
    #     with tempfile.TemporaryDirectory() as root_dir:
    #         outdir = os.path.join(root_dir, 'trees')
    #         phylogeny_generator = PhylogenyGenerator(
    #             a3m_dir_full='test_input_data/a3m_small',
    #             a3m_dir='test_input_data/a3m_small_corrupted',
    #             n_process=n_process,
    #             expected_number_of_MSAs=3,
    #             outdir=outdir,
    #             max_seqs=8,
    #             max_sites=16,
    #             max_families=3,
    #             rate_matrix='None',
    #             fast_tree_cats=20,
    #             use_cached=False,
    #         )
    #         with self.assertRaises(MSAError):
    #             phylogeny_generator.run()

    # @parameterized.expand([("multiprocess", 3), ("single-process", 1)])
    # def test_incorrect_expected_number_of_MSAs_raises_error(self, name, n_process):
    #     """
    #     If the a3m directory has a different number of files from the
    #     expected number, an error should be raised.
    #     """
    #     with tempfile.TemporaryDirectory() as root_dir:
    #         outdir = os.path.join(root_dir, 'trees')
    #         phylogeny_generator = PhylogenyGenerator(
    #             a3m_dir_full='test_input_data/a3m_small',
    #             a3m_dir='test_input_data/a3m_small',
    #             n_process=n_process,
    #             expected_number_of_MSAs=4,
    #             outdir=outdir,
    #             max_seqs=8,
    #             max_sites=16,
    #             max_families=3,
    #             rate_matrix='None',
    #             fast_tree_cats=20,
    #             use_cached=False,
    #         )
    #         with self.assertRaises(ValueError):
    #             phylogeny_generator.run()
