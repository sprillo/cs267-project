from typing import List, Tuple


def write_log_likelihood(
    log_likelihood: Tuple[float, List[float]],
    log_likelihood_path: str
) -> None:
    ll, lls = log_likelihood
    res = ""
    res += f"{ll}\n"
    res += f"{len(lls)} sites\n"
    res += " ".join(lls)
    open(log_likelihood_path, "w").write(res)


def read_log_likelihood(
    log_likelihood_path: str,
) -> Tuple[float, List[float]]:
    lines = open(log_likelihood_path, "r").read().strip().split('\n')
    ll = float(lines[0])
    try:
        num_sites, s = lines[1].split(" ")
        if s != "sites":
            raise Exception
        num_sites = float(num_sites)
    except Exception:
        raise Exception(
            f"Log likelihood file at:{log_likelihood_path} "
            f"should have second line '[num_sites] sites', "
            f"but had second line: {lines[1]} instead."
        )
    lls = list(map(float, lines[2].split(" ")))
    if len(lls) != num_sites:
        raise Exception(
            f"Log likelihood file at:{log_likelihood_path} "
            f"should have {num_sites} values in line 3,"
            f"but had {len(lls)} values instead."
        )
    return ll, lls
