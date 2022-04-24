import logging
import multiprocessing
import os
import tempfile
import time
from typing import List, Tuple

import numpy as np
import pandas as pd
import tqdm
from ete3 import Tree

from src.io import read_rate_matrix
from src.markov_chain import compute_stationary_distribution
from src.utils import get_process_args


def get_rate_categories(
    tree_dir: str,
    protein_family_name: str,
    use_site_specific_rates: bool,
    L: int,
) -> Tuple[List[float], List[int], List[int]]:
    """
    Returns the rates, site_cats, sites_kept for the given protein
    family.

    If use_site_specific_rates=False, it is assumed that all sites evolve
    at the same rate of 1.

    Args:
        tree_dir: Directory where the FastTree log is found.
        protein_family_name: Protein family name.
        use_site_specific_rates: If to use_site_specific_rates. If False, will
            just use a rate of 1 for all sites, which is the old behaviour.
            I.e. we are backwards compatible here.
        L: Number of sites. Used only if use_site_specific_rates=False.
    """
    outlog = os.path.join(tree_dir, protein_family_name + ".log")
    with open(outlog, "r") as outlog_file:
        lines = [line.strip().split() for line in outlog_file]
        if not lines[0][0] == "NCategories":
            raise ValueError(f"NCategories not found in {outlog}")
        ncats = int(lines[0][1])
        if not lines[1][0] == "Rates":
            raise ValueError(f"Rates not found in {outlog}")
        if not len(lines[1][1:]) == ncats:
            raise ValueError(
                f"Rates should be {ncats} in length. Found {len(lines[1][1:])}."
            )
        rates = [float(x) for x in lines[1][1:]]
        if not lines[2][0] == "SiteCategories":
            raise ValueError(f"SiteCategories not found in {outlog}")
        # FastTree uses 1-based indexing for SiteCategories, so we shift by 1.
        site_cats = [int(x) - 1 for x in lines[2][1:]]
    out_sites_kept_path = os.path.join(
        tree_dir, protein_family_name + ".sites_kept"
    )
    with open(out_sites_kept_path, "r") as out_sites_kept_file:
        sites_kept = [
            int(x) for x in out_sites_kept_file.read().strip().split()
        ]
    if not len(site_cats) == len(sites_kept):
        raise ValueError(
            f"SiteCategories should have length {len(sites_kept)},"
            f" but has {len(site_cats)} instead."
        )
    if not use_site_specific_rates:
        # Just force all rates to 1, and use all sites, which is the old
        # behavior. (So, we are backwards compatible.)
        rates = [1.0 for _ in rates]
        site_cats = [0] * L
        sites_kept = list(range(L))
    return rates, site_cats, sites_kept


def get_site_rate_from_site_id(
    site_id: int,
    rates: List[float],
    site_cats: List[int],
    sites_kept: List[int],
):
    """
    Precondition: the site_id must be in the sites_kept, or an error will be
    raised.
    """
    if not len(site_cats) == len(sites_kept):
        raise ValueError(
            f"site_cats and sites_kept should have the same length."
            f" len(site_cats) = {len(site_cats)}; "
            f"len(sites_kept) = {len(sites_kept)}"
        )
    # Find its rate
    for site, site_cat in zip(sites_kept, site_cats):
        if site == site_id:
            return rates[site_cat]
    raise ValueError(
        "Trying to retrieve rate for site that was not kept."
        f" Site: {site_id}. sites_kept = {sites_kept}"
    )


def to_fast_tree_format(rate_matrix: np.array, output_path: str, pi: np.array):
    r"""
    The weird 20 x 21 format of FastTree, which is also column-stochastic.
    """
    amino_acids = [
        "A",
        "R",
        "N",
        "D",
        "C",
        "Q",
        "E",
        "G",
        "H",
        "I",
        "L",
        "K",
        "M",
        "F",
        "P",
        "S",
        "T",
        "W",
        "Y",
        "V",
    ]
    rate_matrix_df = pd.DataFrame(
        rate_matrix, index=amino_acids, columns=amino_acids
    )
    rate_matrix_df = rate_matrix_df.transpose()
    rate_matrix_df["*"] = pi
    with open(output_path, "w") as outfile:
        for aa in amino_acids:
            outfile.write(aa + "\t")
        outfile.write("*\n")
        outfile.flush()
    rate_matrix_df.to_csv(output_path, sep="\t", header=False, mode="a")


def run_fast_tree_with_custom_rate_matrix(
    msa_path: str,
    family: str,
    rate_matrix_path: str,
    num_rate_categories: int,
    output_tree_dir: str,
) -> str:
    r"""
    This wrapper deals with the fact that FastTree only accepts normalized rate
    matrices as input. Therefore, to run FastTree with an arbitrary rate matrix,
    we first have to normalize it. After inference with FastTree, we have to
    'de-normalize' the branch lengths to put them in the same time units as the
    original rate matrix.
    """
    with tempfile.NamedTemporaryFile("w") as scaled_tree_file:
        scaled_tree_filename = (
            scaled_tree_file.name
        )  # Where FastTree will write its output.
        with tempfile.NamedTemporaryFile("w") as scaled_rate_matrix_file:
            scaled_rate_matrix_filename = (
                scaled_rate_matrix_file.name
            )  # The rate matrix for FastTree
            Q_df = read_rate_matrix(rate_matrix_path)
            if not (Q_df.shape == (20, 20)):
                raise ValueError(
                    f"The rate matrix {rate_matrix_path} does not have "
                    "dimension 20 x 20."
                )
            Q = np.array(Q_df)
            pi = compute_stationary_distribution(Q)
            # Check that rows (originally columns) of Q add to 0
            if not np.sum(np.abs(Q.sum(axis=1))) < 0.01:
                raise ValueError(
                    f"Custom rate matrix {rate_matrix_path} doesn't have "
                    "columns that add up to 0."
                )
            # Check that the stationary distro is correct
            if not np.sum(np.abs(pi @ Q)) < 0.01:
                raise ValueError(
                    f"Custom rate matrix {rate_matrix_path} doesn't have the "
                    "stationary distribution."
                )
            # Compute the mutation rate.
            mutation_rate = pi @ -np.diag(Q)
            # Normalize Q
            Q_normalized = Q / mutation_rate
            # Write out Q_normalized in FastTree format, for use in FastTree
            to_fast_tree_format(
                Q_normalized,
                output_path=scaled_rate_matrix_filename,
                pi=pi.reshape(20),
            )
            # Run FastTree!
            dir_path = os.path.dirname(os.path.realpath(__file__))
            outlog = os.path.join(output_tree_dir, family + ".fast_tree_log")
            command = (
                f"{dir_path}/FastTree -quiet -trans "
                + f"{scaled_rate_matrix_filename} -log {outlog} -cat "
                + f"{num_rate_categories} < {msa_path} > "
                + f"{scaled_tree_filename}"
            )
            st = time.time()
            os.system(command)
            et = time.time()
            open(os.path.join(output_tree_dir, family + ".command"), "w").write(
                f"time_fast_tree: {et - st}"
            )
            # De-normalize the branch lengths of the tree
            tree = Tree(scaled_tree_filename)

            def dfs_scale_tree(v: tree) -> None:
                for u in v.get_children():
                    u.dist = u.dist / mutation_rate
                    dfs_scale_tree(u)

            dfs_scale_tree(tree)
            tree.write(
                format=2,
                outfile=os.path.join(output_tree_dir, family + ".newick"),
            )
            open(os.path.join(output_tree_dir, family + ".command"), "w").write(
                command
            )
            # TODO: Convert into our friendy format by using the
            # write_tree function!


def post_process_fast_tree_log(outlog: str):
    """
    We just want the sites and rates, so we prune the FastTree log file to keep
    just this information.
    """
    res = ""
    with open(outlog, "r") as infile:
        for line in infile:
            if (
                line.startswith("NCategories")
                or line.startswith("Rates")
                or line.startswith("SiteCategories")
            ):
                res += line
    with open(outlog, "w") as outfile:
        outfile.write(res)
        outfile.flush()


def _map_func(args: List):
    msa_dir = args[0]
    families = args[1]
    rate_matrix_path = args[2]
    num_rate_categories = args[3]
    output_tree_dir = args[4]

    for family in families:
        msa_path = os.path.join(msa_dir, family + ".txt")
        run_fast_tree_with_custom_rate_matrix(
            msa_path=msa_path,
            family=family,
            rate_matrix_path=rate_matrix_path,
            num_rate_categories=num_rate_categories,
            output_tree_dir=output_tree_dir,
        )


# @cached_parallel_computation(
#     exclude_args=["num_processes"],
#     parallel_arg="families",
#     output_dirs=["output_tree_dir"],
# )
def fast_tree(
    msa_dir: str,
    families: List[str],
    rate_matrix_path: str,
    num_rate_categories: int,
    output_tree_dir: str,
    num_processes: int,
) -> None:
    logger = logging.getLogger("rate_estimation.fast_tree")

    if not os.path.exists(output_tree_dir):
        os.makedirs(output_tree_dir)

    dir_path = os.path.dirname(os.path.realpath(__file__))
    c_path = os.path.join(dir_path, "FastTree.c")
    bin_path = os.path.join(dir_path, "FastTree")
    if not os.path.exists(bin_path):
        os.system(
            "wget http://www.microbesonline.org/fasttree/FastTree.c -P "
            f"{dir_path}"
        )
        compile_command = (
            "gcc -DNO_SSE -DUSE_DOUBLE -O3 -finline-functions -funroll-loops"
            + f" -Wall -o {bin_path} {c_path} -lm"
        )
        logger.info(f"Compiling FastTree with:\n{compile_command}")
        # See http://www.microbesonline.org/fasttree/#Install
        os.system(compile_command)
        if not os.path.exists(bin_path):
            raise Exception("Was not able to compile FastTree")

    map_args = [
        [
            msa_dir,
            get_process_args(process_rank, num_processes, families),
            rate_matrix_path,
            num_rate_categories,
            output_tree_dir,
        ]
        for process_rank in range(num_processes)
    ]

    if num_processes > 1:
        with multiprocessing.Pool(num_processes) as pool:
            list(tqdm.tqdm(pool.imap(_map_func, map_args), total=len(map_args)))
    else:
        list(tqdm.tqdm(map(_map_func, map_args), total=len(map_args)))
