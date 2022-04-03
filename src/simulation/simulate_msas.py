import os
from typing import Dict, List

import hashlib
import numpy as np

from src.io import read_tree, read_site_rates, read_contact_map, read_rate_matrix, read_probability_distribution, write_msa


def sample(
    probability_distribution: np.array
) -> int:
    return np.random.choice(range(len(probability_distribution)), p=probability_distribution)


def sample_transition(
    starting_state: int,
    rate_matrix: np.array,
    elapsed_time: float,
    strategy: str,
):
    """
    Sample the ending state of the Markov chain.

    Args:
        starting_state: The starting state (an integer)
        rate_matrix: The rate matrix
        elapsed_time: The amount of time that the chain is run.
        strategy: Either "all_transitions" or "node_states".
            The "node_states" strategy uses the matrix exponential to figure
            out how to change state from node to node directly. The
            "all_transitions" strategy samples waiting times from an exponential
            distribution to figure out _all_ changes in the elapsed time and
            thus get the ending state. (It is unclear which method will be
            faster)
    """
    return 0


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
    random_seed: int,
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
        random_seed: Random seed for reproducibility. Using the same random
            seed and strategy leads to the exact same simulated data.
        num_processes: Number of processes used to parallelize computation.
    """
    for family in families:
        tree = read_tree(tree_path=os.path.join(tree_dir, family + ".txt"))
        site_rates = read_site_rates(site_rates_path=os.path.join(site_rates_dir, family + ".txt"))
        contact_map = read_contact_map(contact_map_path=os.path.join(contact_map_dir, family + ".txt"))
        pi_1 = read_probability_distribution(pi_1_path).to_numpy().reshape(-1)
        Q_1 = read_rate_matrix(Q_1_path).to_numpy()
        pi_2 = read_probability_distribution(pi_2_path).to_numpy().reshape(-1)
        Q_2 = read_rate_matrix(Q_2_path).to_numpy()

        num_sites = len(site_rates)

        contacting_pairs = list(zip(*np.where(contact_map == 1)))
        contacting_pairs = [
            (i, j)
            for (i, j) in contacting_pairs
            if i < j
        ]
        # Validate that each site is in contact with at most one other site
        contacting_sites = list(sum(contacting_pairs, ()))
        if len(set(contacting_sites)) != len(contacting_sites):
            raise Exception(
                f"Each site can only be in contact with one other site. "
                f"The contacting sites were: {contacting_pairs}"
            )
        independent_sites = [i for i in range(num_sites) if i not in contacting_sites]

        n_independent_sites = len(independent_sites)
        n_contacting_pairs = len(contacting_pairs)

        # We work with *integer states* and then convert back to amino acids at
        # the end. The first n_independent_sites columns will evolve
        # independently, and the last n_independent_sites columns will
        # co-evolve.
        seed = int(hashlib.md5(family.encode()).hexdigest()[:8], 16) + random_seed
        np.random.seed(seed)
        msa_int = {}  # type: Dict[str, List[int]]

        # Sample root state
        root_states = []
        for i in range(n_independent_sites):
            root_states.append(sample(pi_1))
        for i in range(n_contacting_pairs):
            root_states.append(sample(pi_2))
        msa_int[tree.root()] = root_states

        # Depth first search from root
        for node in tree.preorder_traversal():
            if node == tree.root():
                continue
            node_states_int = []
            parent, branch_length = tree.parent(node)
            parent_states = msa_int[parent]
            for i in range(n_independent_sites):
                node_states_int.append(
                    sample_transition(
                        starting_state=parent_states[i],
                        rate_matrix=Q_1,
                        elapsed_time=branch_length * site_rates[independent_sites[i]],
                        strategy=strategy,
                    )
                )
            for i in range(n_contacting_pairs):
                node_states_int.append(
                    sample_transition(
                        starting_state=parent_states[n_independent_sites + i],
                        rate_matrix=Q_2,
                        elapsed_time=branch_length,  # No adjustment for site rates
                        strategy=strategy,
                    )
                )
            msa_int[node] = node_states_int

        # Now just map back the integer states to amino acids
        msa = {}
        pairs_of_amino_acids = [
            aa1 + aa2 for aa1 in amino_acids for aa2 in amino_acids
        ]
        for node in msa_int.keys():
            node_states_int = msa_int[node]
            node_states = ["" for i in range(num_sites)]
            for i in range(n_independent_sites):
                state_int = node_states_int[i]
                state_str = amino_acids[state_int]
                node_states[independent_sites[i]] = state_str
            for i in range(n_contacting_pairs):
                state_int = node_states_int[n_independent_sites + i]
                state_str = pairs_of_amino_acids[state_int]
                (site_1, site_2) = contacting_pairs[i]
                node_states[site_1] = state_str[0]
                node_states[site_2] = state_str[1]
            msa[node] = ''.join(node_states)
            if not all([state != "" for state in state_str]):
                raise Exception("Error mapping integer states to amino acids.")
        msa_path = os.path.join(output_msa_dir, family + ".txt")
        write_msa(
            msa,
            msa_path,
        )

        # os.system(f'chmod 444 "{msa_path}"')
        # success_path = os.path.join(output_msa_dir, family + ".success")
        # with open(success_path, "w") as success_file:
        #     success_file.write("SUCCESS\n")
