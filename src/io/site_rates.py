from typing import List


def read_site_rates(site_rates_path: str) -> List[float]:
    try:
        return list(
            map(float, open(site_rates_path, "r").read().strip().split(" "))
        )
    except Exception:
        raise Exception(f"Could nor read site rates in file: {site_rates_path}")
