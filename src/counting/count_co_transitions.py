import multiprocessing
import os
from typing import List

import numpy as np
import pandas as pd
import tqdm

from src.io import read_msa, read_tree, write_count_matrices
from src.io.contact_map import read_contact_map
from src.utils import get_process_args, quantize


def _map_func(args):
    """
    Version of count_co_transitions run by an individual process.

    Results from each process are later aggregated in the master process.
    """
    tree_dir = args[0]
    msa_dir = args[1]
    contact_map_dir = args[2]
    families = args[3]
    amino_acids = args[4]
    quantization_points = args[5]
    edge_or_cherry = args[6]
    minimum_distance_for_nontrivial_contact = args[7]

    pairs_of_amino_acids = [
        aa1 + aa2 for aa1 in amino_acids for aa2 in amino_acids
    ]
    num_amino_acids = len(amino_acids)
    count_matrices = {
        q: pd.DataFrame(
            np.zeros(shape=(num_amino_acids**2, num_amino_acids**2)),
            index=pairs_of_amino_acids,
            columns=pairs_of_amino_acids,
        )
        for q in quantization_points
    }
    for family in families:
        tree = read_tree(tree_path=os.path.join(tree_dir, family + ".txt"))
        msa = read_msa(msa_path=os.path.join(msa_dir, family + ".txt"))
        contact_map = read_contact_map(
            contact_map_path=os.path.join(contact_map_dir, family + ".txt")
        )
        contacting_pairs = list(zip(*np.where(contact_map == 1)))
        contacting_pairs = [
            (i, j)
            for (i, j) in contacting_pairs
            if abs(i - j) >= minimum_distance_for_nontrivial_contact and i < j
        ]
        for node in tree.nodes():
            node_seq = msa[node]
            if edge_or_cherry == "edge":
                # Extract all transitions on edges starting at 'node'
                for (child, branch_length) in tree.children(node):
                    child_seq = msa[child]
                    q = quantize(branch_length, quantization_points)
                    if q is not None:
                        for (i, j) in contacting_pairs:
                            start_state = node_seq[i] + node_seq[j]
                            end_state = child_seq[i] + child_seq[j]
                            if (
                                node_seq[i] in amino_acids
                                and node_seq[j] in amino_acids
                                and child_seq[i] in amino_acids
                                and child_seq[j] in amino_acids
                            ):
                                count_matrices[q].loc[
                                    start_state, end_state
                                ] += 0.5
                                count_matrices[q].loc[
                                    start_state[::-1], end_state[::-1]
                                ] += 0.5
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
                    branch_length_total = branch_length_1 + branch_length_2
                    q = quantize(branch_length_total, quantization_points)
                    if q is not None:
                        for (i, j) in contacting_pairs:
                            # We accumulate the transitions in both directions
                            start_state = leaf_seq_1[i] + leaf_seq_1[j]
                            end_state = leaf_seq_2[i] + leaf_seq_2[j]
                            if (
                                leaf_seq_1[i] in amino_acids
                                and leaf_seq_1[j] in amino_acids
                                and leaf_seq_2[i] in amino_acids
                                and leaf_seq_2[j] in amino_acids
                            ):
                                count_matrices[q].loc[
                                    start_state, end_state
                                ] += 0.25
                                count_matrices[q].loc[
                                    start_state[::-1], end_state[::-1]
                                ] += 0.25
                                count_matrices[q].loc[
                                    end_state, start_state
                                ] += 0.25
                                count_matrices[q].loc[
                                    end_state[::-1], start_state[::-1]
                                ] += 0.25
    return count_matrices


def count_co_transitions(
    tree_dir: str,
    msa_dir: str,
    contact_map_dir: str,
    families: List[str],
    amino_acids: List[str],
    quantization_points: List[float],
    edge_or_cherry: bool,
    minimum_distance_for_nontrivial_contact: int,
    output_count_matrices_dir: str,
    num_processes: int,
) -> None:
    """
    Count the number of co-transitions.

    For a tree, an MSA, and a contact map, count the number of co-transitions
    between pairs of non-trivial contacting amino acids at cherries of the
    trees. This computation is aggregated over all the families.

    The computational complexity of this function is as follows. Let:
    - f be the number of families,
    - n be the (average) number of sequences in each family,
    - l be the (average) length of each protein,
    - b be the number of quantization points ('buckets'), and
    - s = len(amino_acids) be the number of amino acids ('states'),
    Then the computational complexity is: O(f * (n * l + b * s^4)).

    Details:
    - Branches whose lengths are smaller than the smallest quantization point,
        or larger than the larger quantization point, are ignored.
    - Only transitions involving valid amino acids are counted.
    - A contact between positions (i, j) is considered non-trivial if
        |i - j| >= minimum_distance_for_nontrivial_contact. Otherwise,
        the contact is considered 'trivial' and ignored when counting.

    Args:
        tree_dir: Directory to the trees stored in friendly format.
        msa_dir: Directory to the multiple sequence alignments in FASTA format.
        contact_map_dir: Directory to the contact maps stored as
            space-separated binary matrices.
        families: The protein families for which to perform the computation.
        amino_acids: The list of (valid) amino acids.
        quantization_points: List of time values used to approximate branch
            lengths.
        edge_or_cherry: Whether to count transitions on edges (which are
            unidirectional), or on cherries (which are bi-directional).
        minimum_distance_for_nontrivial_contact: The minimum distance - in
            terms of indexing - required for an amino acid contact to be
            considered non-trivial.
        output_count_matrices_dir: Directory where to write the count matrices
            to.
        num_processes: Number of processes used to parallelize computation.
    """
    map_args = [
        [
            tree_dir,
            msa_dir,
            contact_map_dir,
            get_process_args(process_rank, num_processes, families),
            amino_acids,
            quantization_points,
            edge_or_cherry,
            minimum_distance_for_nontrivial_contact,
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
