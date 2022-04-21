import os
from typing import List

import networkx as nx
import numpy as np

from src.io import read_contact_map, write_contact_map


def create_maximal_matching_contact_map(
    i_contact_map_dir: str,
    families: List[str],
    o_contact_map_dir: str,
    num_processes: int,
) -> None:
    for family in families:
        topology = nx.Graph()
        i_contact_map_path = os.path.join(i_contact_map_dir, family + ".txt")
        contact_map = read_contact_map(contact_map_path=i_contact_map_path)
        num_sites = contact_map.shape[0]
        topology.add_nodes_from([i for i in range(num_sites)])
        contacting_pairs = list(zip(*np.where(contact_map == 1)))
        contacting_pairs = [(i, j) for (i, j) in contacting_pairs if i < j]
        topology.add_edges_from(contacting_pairs)
        print(topology)
        match = nx.maximal_matching(topology)
        res = np.zeros(shape=contact_map.shape)
        for (u, v) in match:
            res[u, v] = res[v, u] = 1
        o_contact_map_path = os.path.join(o_contact_map_dir, family + ".txt")
        write_contact_map(res, o_contact_map_path)
