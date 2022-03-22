from typing import List


def simulate_msas(
    newick_tree_paths: List[str],
    site_rates_paths: List[str],
    contact_map_paths: List[str],
    amino_acids: List[str],
    pi_1_path: str,
    Q_1_path: str,
    pi_2_path: str,
    Q_2_path: str,
    strategy: str,
    output_msa_paths: List[str],
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
    - The Q_2 matrix is sparse: only 2 * len(amino_acids) - 1 entries in each
        row are non-zero, since only one amino acid in a contacting pair
        can mutate at a time.

    Args:
        newick_tree_paths: Paths to the trees stored in newick format.
        site_rates_paths: Paths to the files containing the rates at which
            each site evolves, in the FastTree output format. Rates for sites
            that co-evolve are ignored.
        contact_map_paths: Paths to the contact maps stored as space-separated
            binary matrices.
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
        strategy: Either 'all_transitions' or 'chain_jump'. The
            'all_transitions' strategy samples all state changes on the tree
            and does not require the matrix exponential, while the 'chain_jump'
            strategy does not sample all state changes on the tree but requires
            the matrix exponential.
        output_msa_paths: Paths where to write the multiple sequence alignments
            to in FASTA format.
        num_processes: Number of processes used to parallelize computation.
    """
    pass
