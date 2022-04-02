import os
from typing import List

import numpy as np
import pandas as pd

from src.io import read_tree, read_msa, read_site_rates, write_count_matrices
from src.utils import quantize


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
    length = len(amino_acids)
    count_matices = {q: pd.DataFrame(np.zeros(shape=(length, length)), index=amino_acids, columns=amino_acids) for q in quantization_points}
    for family in families:
        tree = read_tree(tree_path=os.path.join(tree_dir, family + ".txt"))
        msa = read_msa(msa_path=os.path.join(msa_dir, family + ".txt"))
        site_rates = read_site_rates(site_rates_path=os.path.join(site_rates_dir, family + ".txt"))
        for node in tree.nodes():
            node_seq = msa[node]
            msa_length = len(node_seq)
            if edge_or_cherry == "edge":
                # Extract all transitions on edges starting at 'node'
                for (child, branch_length) in tree.children(node):
                    child_seq = msa[child]
                    for amino_acid_idx in range(msa_length):
                        site_rate = site_rates[amino_acid_idx]
                        q = quantize(branch_length * site_rate, quantization_points)
                        if q is not None:
                            start_state = node_seq[amino_acid_idx]
                            end_state = child_seq[amino_acid_idx]
                            if start_state in amino_acids and end_state in amino_acids:
                                count_matices[q].loc[start_state, end_state] += 1
            elif edge_or_cherry == "cherry":
                raise NotImplementedError
    write_count_matrices(count_matices, os.path.join(output_count_matrices_dir, "result.txt"))
