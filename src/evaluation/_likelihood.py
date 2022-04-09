"""
To test this, compare the single-site likelihood against what is obtained from
FastTree. Then, we will change to co-evolution model, and see how much the
likelihood improves. To test the co-evolution computation, we compare it
against FastTree ran with 1 site rate category only (i.e. no site rates)
and a product rate matrix.
"""
from typing import List


def compute_log_likelihoods(
    tree_dir: str,
    msa_dir: str,
    site_rates_dir: str,
    contact_map_dir: str,
    families: List[str],
    amino_acids: List[str],
    pi_1_path: str,
    Q_1_path: str,
    pi_2_path: str,
    Q_2_path: str,
    output_likelihood_dir: str,
    num_processes: int,
    use_cpp_implementation: bool = False,
) -> None:
    """
    Compute log-likelihoods under the given model.

    Given trees, MSAs, site rates, contact maps, and models for the evolution
    of contacting sites and non-contacting sites, the log-likelihood of each
    tree is computed, at the resolution of single-sites and contacting pairs.
    The model is given by the state distribution of the root node, and the rate
    matrix which describes the evolution of states.

    Details:
    - For each position, it must be either in contact with exactly 1 other
        position, or not be in contact with any other position. The diagonal
        of the contact matrix is ignored.
    - The Q_2 matrix is sparse: only 2 * len(amino_acids) - 1 entries in each
        row are non-zero, since only one amino acid in a contacting pair
        can mutate at a time.

    Args:
        tree_dir: Directory to the trees stored in friendly format.
        msa_dir: Directory to the multiple sequence alignments in FASTA format.
        site_rates_dir: Directory to the files containing the rates at which
            each site evolves. Rates for sites that co-evolve are ignored.
        contact_map_dir: Directory to the contact maps stored as
            space-separated binary matrices.
        families: The protein families for which to perform the computation.
        amino_acids: The list of amino acids.
        pi_1_path: Path to an array of length len(amino_acids). It indicates,
            for sites that evolve independently (i.e. that are not in contact
            with any other site), the probabilities for the root state.
        Q_1_path: Path to an array of size len(amino_acids) x len(amino_acids),
            the rate matrix describing the evolution of sites that evolve
            independently (i.e. that are not in contact with any other site).
        pi_2_path: Path to an array of length len(amino_acids) ** 2. It
            indicates, for sites that are in contact, the probabilities for
            their root state.
        Q_2_path: Path to an array of size (len(amino_acids) ** 2) x
            (len(amino_acids) ** 2), the rate matrix describing the evolution
            of sites that are in contact.
        output_likelihood_dir: Directory where to write the log-likelihoods,
            with site-level resolution.
        num_processes: Number of processes used to parallelize computation.
        use_cpp_implementation: If to use efficient C++ implementation
            instead of Python.
    """
    if use_cpp_implementation:
        raise NotImplementedError

    raise NotImplementedError
