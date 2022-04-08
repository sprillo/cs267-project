import multiprocessing
import os
from typing import List

import numpy as np
import pandas as pd
import tqdm

from src.io import read_msa, read_site_rates, read_tree, write_count_matrices
from src.utils import get_process_args, quantization_idx


def _map_func(args):
    """
    Version of count_transitions run by an individual process.

    Results from each process are later aggregated in the master process.
    """
    tree_dir = args[0]
    msa_dir = args[1]
    site_rates_dir = args[2]
    families = args[3]
    amino_acids = args[4]
    quantization_points = np.array(sorted(args[5]))
    edge_or_cherry = args[6]

    num_amino_acids = len(amino_acids)
    aa_to_int = {
        aa: i for (i, aa) in enumerate(amino_acids)
    }
    count_matrices_numpy = np.zeros(shape=(len(quantization_points), num_amino_acids, num_amino_acids))
    for family in families:
        tree = read_tree(tree_path=os.path.join(tree_dir, family + ".txt"))
        msa = read_msa(msa_path=os.path.join(msa_dir, family + ".txt"))
        site_rates = read_site_rates(
            site_rates_path=os.path.join(site_rates_dir, family + ".txt")
        )
        for node in tree.nodes():
            node_seq = msa[node]
            msa_length = len(node_seq)
            if edge_or_cherry == "edge":
                # Extract all transitions on edges starting at 'node'
                for (child, branch_length) in tree.children(node):
                    child_seq = msa[child]
                    for amino_acid_idx in range(msa_length):
                        site_rate = site_rates[amino_acid_idx]
                        q_idx = quantization_idx(
                            branch_length * site_rate, quantization_points
                        )
                        if q_idx is not None:
                            start_state = node_seq[amino_acid_idx]
                            end_state = child_seq[amino_acid_idx]
                            if (
                                start_state in amino_acids
                                and end_state in amino_acids
                            ):
                                start_state_idx = aa_to_int[start_state]
                                end_state_idx = aa_to_int[end_state]
                                count_matrices_numpy[
                                    q_idx,
                                    start_state_idx,
                                    end_state_idx
                                ] += 1
            elif edge_or_cherry == "cherry":
                children = tree.children(node)
                if len(children) == 2 and all(
                    [tree.is_leaf(child) for (child, _) in children]
                ):
                    (leaf_1, branch_length_1), (leaf_2, branch_length_2) = (
                        children[0],
                        children[1],
                    )
                    leaf_seq_1, leaf_seq_2 = msa[leaf_1], msa[leaf_2]
                    for amino_acid_idx in range(msa_length):
                        site_rate = site_rates[amino_acid_idx]
                        branch_length_total = branch_length_1 + branch_length_2
                        q_idx = quantization_idx(
                            branch_length_total * site_rate, quantization_points
                        )
                        if q_idx is not None:
                            start_state = leaf_seq_1[amino_acid_idx]
                            end_state = leaf_seq_2[amino_acid_idx]
                            if (
                                start_state in amino_acids
                                and end_state in amino_acids
                            ):
                                start_state_idx = aa_to_int[start_state]
                                end_state_idx = aa_to_int[end_state]
                                count_matrices_numpy[
                                    q_idx, start_state_idx, end_state_idx
                                ] += 0.5
                                count_matrices_numpy[
                                    q_idx, end_state_idx, start_state_idx
                                ] += 0.5
    count_matrices = {
        q: pd.DataFrame(
            count_matrices_numpy[q_idx, :, :],
            index=amino_acids,
            columns=amino_acids,
        )
        for (q_idx, q) in enumerate(quantization_points)
    }
    return count_matrices


def count_transitions(
    tree_dir: str,
    msa_dir: str,
    site_rates_dir: str,
    families: List[str],
    amino_acids: List[str],
    quantization_points: List[float],
    edge_or_cherry: bool,
    output_count_matrices_dir: str,
    num_processes: int,
) -> None:
    """
    Count the number of transitions.

    For a tree, an MSA, and site rates, count the number of transitions
    between amino acids at either edges of cherries of the trees. This
    computation is aggregated over all the families.

    The computational complexity of this function is as follows. Let:
    - f be the number of families,
    - n be the (average) number of sequences in each family,
    - l be the (average) length of each protein,
    - b be the number of quantization points ('buckets'), and
    - s = len(amino_acids) be the number of amino acids ('states'),
    Then the computational complexity is: O(f * (n * l + b * s^2)).

    Details:
    - Branches whose lengths are smaller than the smallest quantization point,
        or larger than the larger quantization point, are ignored.
    - Only transitions involving valid amino acids are counted.
    - Branch lengths are adjusted by the site-specific rate when counting.

    Args:
        tree_dir: Directory to the trees stored in friendly format.
        msa_dir: Directory to the multiple sequence alignments in FASTA format.
        site_rates_dir: Directory to the files containing the rates at which
            each site evolves.
        families: The protein families for which to perform the computation.
        amino_acids: The list of (valid) amino acids.
        quantization_points: List of time values used to approximate branch
            lengths.
        edge_or_cherry: Whether to count transitions on edges (which are
            unidirectional), or on cherries (which are bi-directional).
        output_count_matrices_dir: Directory where to write the count matrices
            to.
        num_processes: Number of processes used to parallelize computation.
    """
    map_args = [
        [
            tree_dir,
            msa_dir,
            site_rates_dir,
            get_process_args(process_rank, num_processes, families),
            amino_acids,
            quantization_points,
            edge_or_cherry,
        ]
        for process_rank in range(num_processes)
    ]

    # Map step (distribute families among processes)
    if num_processes > 1:
        with multiprocessing.Pool(num_processes) as pool:
            count_matrices_per_process = list(
                tqdm.tqdm(pool.imap(_map_func, map_args), total=len(map_args))
            )
    else:
        count_matrices_per_process = list(
            tqdm.tqdm(map(_map_func, map_args), total=len(map_args))
        )

    # Reduce step (aggregate count matrices from all processes)
    count_matrices = count_matrices_per_process[0]
    for process_rank in range(1, num_processes):
        for q in quantization_points:
            count_matrices[q] += count_matrices_per_process[process_rank][q]

    write_count_matrices(
        count_matrices, os.path.join(output_count_matrices_dir, "result.txt")
    )
