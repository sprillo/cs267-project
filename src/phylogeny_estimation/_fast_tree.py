import logging
import multiprocessing
import os
import tempfile
import time
from typing import List, Optional

import numpy as np
import pandas as pd
import tqdm
from ete3 import Tree as TreeETE

import src.utils as utils
from src.caching import cached_parallel_computation
from src.io import Tree, read_rate_matrix, write_tree
from src.markov_chain import compute_stationary_distribution
from src.utils import get_process_args


def to_fast_tree_format(rate_matrix: np.array, output_path: str, pi: np.array):
    r"""
    The weird 20 x 21 format of FastTree, which is also column-stochastic.
    """
    amino_acids = utils.amino_acids[:]
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


def name_internal_nodes(t: TreeETE) -> None:
    r"""
    Assigns names to the internal nodes of tree t if they don't already have a
    name.
    """

    def node_name_generator():
        """Generates unique node names for the tree."""
        internal_node_id = 1
        while True:
            yield f"internal-{internal_node_id}"
            internal_node_id += 1

    names = node_name_generator()

    def dfs_name_internal_nodes(p: Optional[Tree], v: Tree) -> None:
        global internal_node_id
        if v.name == "":
            v.name = next(names)
        if p:
            # print(f"{p.name} -> {v.name}")
            pass
        for u in v.get_children():
            dfs_name_internal_nodes(v, u)

    dfs_name_internal_nodes(None, t)


def translate_site_rates(
    i_fasttree_log_dir: str,
    family: str,
    o_site_rates_dir: str,
) -> None:
    lines = (
        open(os.path.join(i_fasttree_log_dir, family + ".fast_tree_log"), "r")
        .read()
        .split("\n")
    )
    for (j, line) in enumerate(lines):
        if line.startswith("Rates"):
            lines_1_split = lines[j].split(" ")
            lines_2_split = lines[j + 1].split(" ")
            site_rates = [
                lines_1_split[int(lines_2_split[i + 1])]
                for i in range(len(lines_2_split) - 1)
            ]
    open(os.path.join(o_site_rates_dir, family + ".txt"), "w").write(
        f"{len(site_rates)} sites\n" + " ".join(site_rates)
    )


def run_fast_tree_with_custom_rate_matrix(
    msa_path: str,
    family: str,
    rate_matrix_path: str,
    num_rate_categories: int,
    output_tree_dir: str,
    output_site_rates_dir: str,
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
            open(
                os.path.join(output_tree_dir, family + ".profiling"), "w"
            ).write(f"time_fast_tree: {et - st}")
            # De-normalize the branch lengths of the tree
            tree_ete = TreeETE(scaled_tree_filename)

            def dfs_scale_tree(v) -> None:
                for u in v.get_children():
                    u.dist = u.dist / mutation_rate
                    dfs_scale_tree(u)

            dfs_scale_tree(tree_ete)

            name_internal_nodes(tree_ete)

            tree_ete.write(
                format=2,
                outfile=os.path.join(output_tree_dir, family + ".newick"),
            )
            open(os.path.join(output_tree_dir, family + ".command"), "w").write(
                command
            )
            tree = Tree()

            def dfs_translate_tree(p, v) -> None:
                tree.add_node(v.name)
                if p is not None:
                    tree.add_edge(p.name, v.name, v.dist)
                for u in v.get_children():
                    dfs_translate_tree(v, u)

            dfs_translate_tree(None, tree_ete)

            write_tree(tree, os.path.join(output_tree_dir, family + ".txt"))

            translate_site_rates(
                i_fasttree_log_dir=output_tree_dir,
                family=family,
                o_site_rates_dir=output_site_rates_dir,
            )


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
    output_site_rates_dir = args[5]

    for family in families:
        msa_path = os.path.join(msa_dir, family + ".txt")
        run_fast_tree_with_custom_rate_matrix(
            msa_path=msa_path,
            family=family,
            rate_matrix_path=rate_matrix_path,
            num_rate_categories=num_rate_categories,
            output_tree_dir=output_tree_dir,
            output_site_rates_dir=output_site_rates_dir,
        )


@cached_parallel_computation(
    exclude_args=["num_processes"],
    parallel_arg="families",
    output_dirs=["output_tree_dir", "output_site_rates_dir"],
)
def fast_tree(
    msa_dir: str,
    families: List[str],
    rate_matrix_path: str,
    num_rate_categories: int,
    output_tree_dir: str,
    output_site_rates_dir: str,
    num_processes: int,
) -> None:
    logger = logging.getLogger("rate_estimation.fast_tree")

    if not os.path.exists(output_tree_dir):
        os.makedirs(output_tree_dir)
    if not os.path.exists(output_site_rates_dir):
        os.makedirs(output_site_rates_dir)

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
            output_site_rates_dir,
        ]
        for process_rank in range(num_processes)
    ]

    if num_processes > 1:
        with multiprocessing.Pool(num_processes) as pool:
            list(tqdm.tqdm(pool.imap(_map_func, map_args), total=len(map_args)))
    else:
        list(tqdm.tqdm(map(_map_func, map_args), total=len(map_args)))
