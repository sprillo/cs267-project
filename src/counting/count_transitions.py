from typing import List


def count_transitions(
    newick_tree_paths: List[str],
    msa_paths: List[str],
    site_rates_paths: List[str],
    amino_acids: List[str],
    quantization_points: List[float],
    output_count_matrices_path: str,
    num_processes: int,
):
    """
    Count the number of transitions.

    For a given list of protein family trees, MSAs, and site rates, count
    the number of transitions between amino acids at cherries of the trees.

    The computational complexity of this function is as follows. Let:
    - t be the number of trees,
    - n be the (average) number of sequences in each tree,
    - l be the (average) length of each protein,
    - b be the number of quantization points ('buckets'), and
    - s = len(amino_acids) be the number of amino acids ('states'),
    Then the computational complexity is: O(t * (n * l + b * s^2)).

    Details:
    - Branches whose lengths are smaller than the smallest quantization point,
        or larger than the larger quantization point, are ignored.
    - Only transitions involving valid amino acids are counted.
    - Branch lengths are adjusted by the site-specific rate when counting.

    Args:
        newick_tree_paths: Paths to the trees stored in newick format.
        msa_paths: Paths to the multiple sequence alignments in FASTA format.
        site_rates_paths: Paths to the files containing the rates at which
            each site evolves, in the FastTree output format.
        amino_acids: The list of (valid) amino acids.
        quantization_points: List of time values used to approximate branch
            lengths.
        output_count_matrices_path: Path where to write the count matrices to.
        num_processes: Number of processes used to parallelize computation.
    """
    pass
