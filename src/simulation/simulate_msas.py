from typing import List


def simulate_msas(
    tree_dir: str,
    site_rates_dir: str,
    contact_map_dir: str,
    families: List[str],
    amino_acids: List[str],
    pi_1_path: str,
    Q_1_path: str,
    pi_2_path: str,
    Q_2_path: str,
    strategy: str,
    output_msa_dir: str,
    num_processes: int,
) -> None:
    """
    Simulate multiple sequence alignments (MSAs).

    Given a contact map and models for the evolution of contacting sites and
    non-contacting sites, protein sequences are simulated and written out to
    output_msa_paths.

    Details:
    - For each position, it must be either in contact with exactly 1 other
        position, or not be in contact with any other position. The diagonal
        of the contact matrix is ignored.
    - If i is in contact with j, then j is in contact with i (i.e. the relation
        is symmetric, and so the contact map is symmetric).
    - The Q_2 matrix is sparse: only 2 * len(amino_acids) - 1 entries in each
        row are non-zero, since only one amino acid in a contacting pair
        can mutate at a time.

    Args:
        tree_dir: Directory to the trees stored in friendly format.
        site_rates_dir: Directory to the files containing the rates at which
            each site evolves. Rates for sites that co-evolve are ignored.
        contact_map_dir: Directory to the contact maps stored as
            space-separated binary matrices.
        families: The protein families for which to perform the computation.
        amino_acids: The list of (valid) amino acids.
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
        strategy: Either 'all_transitions' or 'chain_jump'. The
            'all_transitions' strategy samples all state changes on the tree
            and does not require the matrix exponential, while the 'chain_jump'
            strategy does not sample all state changes on the tree but requires
            the matrix exponential.
        output_msa_dir: Directory where to write the multiple sequence
            alignments to in FASTA format.
        num_processes: Number of processes used to parallelize computation.
    """
    pass
