from typing import List


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
        contact_map_paths: Directory to the contact maps stored as
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
    pass
