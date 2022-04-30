import os
import time
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import src.caching as caching
from src.io import read_msa, write_contact_map, write_msa


@caching.cached()
def get_families(
    msa_dir: str,
) -> List[str]:
    """
    Get the list of protein families names.

    Args:
        msa_dir: Directory with the MSA files. There should be one file with
            name family.txt for each protein family.

    Returns:
        The list of protein family names in the provided directory.
    """
    families = sorted(list(os.listdir(msa_dir)))
    families = [x.split(".")[0] for x in families if x.endswith(".txt")]
    return families


@caching.cached()
def get_family_sizes(
    msa_dir: str,
    max_families: Optional[int] = None,
) -> pd.DataFrame:
    """
    Get the size of each protein family.

    By 'size' we mean the number of sequences and the number of sites. These
    are returned in a Pandas DataFrame object, with one row per family.

    Args:
        msa_dir: Directory with the MSA files. There should be one file with
            name family.txt for each protein family.
        max_families: Only return the results for the first `max_families`
            families in lexicographic order.

    Returns:
        A Pandas DataFrame with one row per family, containing the num_sequences
        and num_sites.
    """
    families = get_families(msa_dir=msa_dir)
    if max_families is None:
        max_families = len(families)
    family_size_tuples = []
    for family in families[:max_families]:
        msa = read_msa(os.path.join(msa_dir, f"{family}.txt"))
        num_sequences = len(msa)
        num_sites = len(next(iter(msa.values())))
        family_size_tuples.append((family, num_sequences, num_sites))
    family_size_df = pd.DataFrame(
        family_size_tuples, columns=["family", "num_sequences", "num_sites"]
    )
    return family_size_df


def get_families_within_cutoff(
    msa_dir: str,
    min_num_sites: int,
    max_num_sites: int,
    min_num_sequences: int,
    max_num_sequences: int,
) -> List[str]:
    family_size_df = get_family_sizes(
        msa_dir=msa_dir,
        max_families=None,
    )
    families = family_size_df.family[
        (family_size_df.num_sites >= min_num_sites)
        & (family_size_df.num_sites <= max_num_sites)
        & (family_size_df.num_sequences >= min_num_sequences)
        & (family_size_df.num_sequences <= max_num_sequences)
    ]
    families = list(families)
    return families


def fig_family_sizes(
    msa_dir: str,
    max_families: Optional[int] = None,
) -> None:
    """
    Histograms of num_sequences and num_sites.
    """
    family_size_df = get_family_sizes(
        msa_dir=msa_dir,
        max_families=max_families,
    )
    plt.title("Distribution of family num_sequences")
    plt.hist(family_size_df.num_sequences)
    plt.show()

    plt.title("Distribution of family num_sites")
    print(f"median num_sites = {family_size_df.num_sites.median()}")
    print(f"mode num_sites = {family_size_df.num_sites.mode()}")
    print(f"mean num_sites = {family_size_df.num_sites.mean()}")
    plt.hist(family_size_df.num_sites, bins=100)
    plt.show()

    plt.xlabel("num_sequences")
    plt.ylabel("num_sites")
    plt.scatter(
        family_size_df.num_sequences, family_size_df.num_sites, alpha=0.3
    )
    plt.show()


@caching.cached_parallel_computation(
    parallel_arg="families",
    output_dirs=["output_contact_map_dir"],
)
def create_trivial_contact_maps(
    msa_dir: str,
    families: List[str],
    states: List[str],
    output_contact_map_dir: str,
):
    for family in families:
        st = time.time()
        msa = read_msa(os.path.join(msa_dir, family + ".txt"))
        num_sites = len(next(iter(msa.values())))
        contact_map = np.zeros(shape=(num_sites, num_sites), dtype=int)
        write_contact_map(
            contact_map, os.path.join(output_contact_map_dir, family + ".txt")
        )
        et = time.time()
        open(
            os.path.join(output_contact_map_dir, family + ".profiling"), "w"
        ).write(f"Total time: {et - st}\n")


@caching.cached_parallel_computation(
    parallel_arg="families",
    output_dirs=["output_msa_dir"],
)
def subset_msa_to_leaf_nodes(
    msa_dir: str,
    families: List[str],
    states: List[str],
    output_msa_dir: str,
):
    """
    An internal node is anyone that starts with 'internal-'.
    """
    for family in families:
        msa = read_msa(os.path.join(msa_dir, family + ".txt"))
        msa_subset = {
            seq_name: seq
            for (seq_name, seq) in msa.items()
            if not seq_name.startswith("internal-")
        }
        write_msa(msa_subset, os.path.join(output_msa_dir, family + ".txt"))
