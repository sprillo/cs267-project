import hashlib
import multiprocessing
import os
from typing import List, Optional, Tuple

import numpy as np
import tqdm

from src import caching
from src.io import write_msa
from src.utils import get_process_args


def subsample_pfam_15k_msa(
    input_msa_path: str,
    num_sequences: Optional[int],
    output_msa_dir: str,
    family: str,
):
    if not os.path.exists(input_msa_path):
        raise FileNotFoundError(f"MSA file {input_msa_path} does not exist!")

    # Read MSA
    msa = []  # type: List[Tuple[str, str]]
    with open(input_msa_path) as file:
        lines = list(file)
        n_lines = len(lines)
        for i in range(0, n_lines, 2):
            if not lines[i][0] == ">":
                raise Exception("Protein name line should start with '>'")
            protein_name = lines[i][1:].strip()
            protein_seq = lines[i + 1].strip()
            # Lowercase amino acids in the sequence are repetitive
            # sequences and should be ignored.
            protein_seq = "".join([c for c in protein_seq if not c.islower()])
            msa.append((protein_name, protein_seq))
        # Check that all sequences in the MSA have the same length.
        for i in range(len(msa) - 1):
            if len(msa[i][1]) != len(msa[i + 1][1]):
                raise Exception(
                    f"Sequence\n{msa[i][1]}\nand\n{msa[i + 1][1]}\nin the "
                    f"MSA do not have the same length! ({len(msa[i][1])} vs"
                    f" {len(msa[i + 1][1])})"
                )

    # Subsample MSA
    family_hash = (
        int(hashlib.sha512(family.encode("utf-8")).hexdigest(), 16) % 10**8
    )
    rng = np.random.default_rng(family_hash)
    nseqs = len(msa)
    if num_sequences is not None:
        max_seqs = min(nseqs, num_sequences)
        seqs_to_keep = [0] + list(
            rng.choice(range(1, nseqs, 1), size=max_seqs - 1, replace=False)
        )
        seqs_to_keep = sorted(seqs_to_keep)
        msa = [msa[i] for i in seqs_to_keep]
    msa_dict = dict(msa)
    write_msa(
        msa=msa_dict, msa_path=os.path.join(output_msa_dir, family + ".txt")
    )


def _map_func(args: List):
    msa_dir = args[0]
    num_sequences = args[1]
    families = args[2]
    output_msa_dir = args[3]
    for family in families:
        subsample_pfam_15k_msa(
            input_msa_path=os.path.join(msa_dir, family + ".a3m"),
            num_sequences=num_sequences,
            output_msa_dir=output_msa_dir,
            family=family,
        )


@caching.cached_parallel_computation(
    exclude_args=["num_processes"],
    parallel_arg="families",
    output_dirs=["output_msa_dir"],
)
def subsample_pfam_15k_msas(
    msa_dir: str,
    num_sequences: int,
    families: List[str],
    output_msa_dir: str,
    num_processes: int,
):
    map_args = [
        [
            msa_dir,
            num_sequences,
            get_process_args(process_rank, num_processes, families),
            output_msa_dir,
        ]
        for process_rank in range(num_processes)
    ]

    if num_processes > 1:
        with multiprocessing.Pool(num_processes) as pool:
            list(tqdm.tqdm(pool.imap(_map_func, map_args), total=len(map_args)))
    else:
        list(tqdm.tqdm(map(_map_func, map_args), total=len(map_args)))
