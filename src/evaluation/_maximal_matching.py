import os
from typing import List

import networkx as nx
import numpy as np

from src.caching import cached_parallel_computation
from src.io import read_contact_map, write_contact_map


@cached_parallel_computation(
    exclude_args=["num_processes"],
    parallel_arg="families",
    output_dirs=["o_contact_map_dir"],
)
def create_maximal_matching_contact_map(
    i_contact_map_dir: str,
    families: List[str],
    o_contact_map_dir: str,
    minimum_distance_for_nontrivial_contact: int,
    num_processes: int,
) -> None:
    if num_processes != 1:
        raise NotImplementedError("Multiprocessing not yet implemented.")
    for family in families:
        topology = nx.Graph()
        i_contact_map_path = os.path.join(i_contact_map_dir, family + ".txt")
        contact_map = read_contact_map(contact_map_path=i_contact_map_path)
        num_sites = contact_map.shape[0]
        topology.add_nodes_from([i for i in range(num_sites)])
        contacting_pairs = list(zip(*np.where(contact_map == 1)))
        contacting_pairs = [
            (i, j)
            for (i, j) in contacting_pairs
            if i < j and abs(i - j) >= minimum_distance_for_nontrivial_contact
        ]
        topology.add_edges_from(contacting_pairs)
        match = nx.maximal_matching(topology)
        res = np.zeros(shape=contact_map.shape)
        for (u, v) in match:
            res[u, v] = res[v, u] = 1
        o_contact_map_path = os.path.join(o_contact_map_dir, family + ".txt")
        write_contact_map(res, o_contact_map_path)
