from typing import List
import numpy as np


def count_co_transitions(
    newick_tree_paths: List[str],
    msa_paths: List[str],
    contact_map_paths: List[str],
    amino_acids: List[str],
    quantization_points: List[float],
    minimum_distance_for_nontrivial_contact: int,
    num_processes: int,
) -> List[np.array]:
    """
    Count the number of co-transitions.

    For a given list of protein family trees, MSAs, and contact maps, count
    the number of co-transitions between pairs of non-trivial contacting amino
    acids at cherries of the trees.

    The computational complexity of this function is as follows. Let:
    - t be the number of trees,
    - n be the (average) number of sequences in each tree,
    - l be the (average) length of each protein,
    - b be the number of quantization points ('buckets'), and
    - s = len(amino_acids) be the number of amino acids ('states'),
    Then the computational complexity is: O(t * (n * l + b * s^4)).

    Details:
    - Branches whose lengths are smaller than the smallest quantization point,
        or larger than the larger quantization point, are ignored.
    - Only transitions involving valid amino acids are counted.
    - A contact between positions (i, j) is considered non-trivial if
        |i - j| >= minimum_distance_for_nontrivial_contact. Otherwise,
        the contact is considered 'trivial' and ignored when counting.

    Args:
        newick_tree_paths: Paths to the trees stored in newick format.
        msa_paths: Paths to the multiple sequence alignments in FASTA format.
        contact_map_paths: Paths to the contact maps stored as space-separated
            binary matrices.
        amino_acids: The list of (valid) amino acids.
        quantization_points: List of time values used to approximate branch
            lengths.
        minimum_distance_for_nontrivial_contact: The minimum distance - in
            terms of indexing - required for an amino acid contact to be
            considered non-trivial.
        num_processes: Number of processes used to parallelize computation.

    Returns:
        An np.array of size b x s^4. Is is the list of co-transition count
            matrices, one for each quantization point.
    """
    pass
