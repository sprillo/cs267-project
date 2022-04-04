import os

from typing import List


def read_site_rates(site_rates_path: str) -> List[float]:
    try:
        return list(
            map(float, open(site_rates_path, "r").read().strip().split(" "))
        )
    except Exception:
        raise Exception(f"Could nor read site rates in file: {site_rates_path}")


def write_site_rates(
    site_rates: List[float],
    site_rates_path: str
) -> None:
    site_rates_dir = os.path.dirname(site_rates_path)
    if not os.path.exists(site_rates_dir):
        os.makedirs(site_rates_dir)
    res = " ".join(list(map(str, site_rates)))
    with open(site_rates_path, "w") as outfile:
        outfile.write(res)
        outfile.flush()
