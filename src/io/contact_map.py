import numpy as np


def read_contact_map(
    contact_map_path: str,
) -> np.array:
    lines = open(contact_map_path).read().strip().split("\n")
    try:
        num_sites, s = lines[0].split(" ")
        if s != "sites":
            raise Exception
        num_sites = int(num_sites)
    except Exception:
        raise Exception(
            f"Contact map file should start with line '[num_sites] sites', but started with: {lines[0]} instead."
        )
    res = np.zeros(shape=(num_sites, num_sites), dtype=np.int)
    for i in range(num_sites):
        res[i, :] = np.array(list(map(int, lines[i + 1].split())))
    return res
