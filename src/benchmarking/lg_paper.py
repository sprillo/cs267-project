import logging
import multiprocessing
import os
import sys
import tempfile
from typing import Callable, Dict, List, Optional, Tuple

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tqdm
import wget

from src import caching
from src.phylogeny_estimation import phyml
from src.utils import pushd

dir_path = os.path.dirname(os.path.realpath(__file__))


def init_logger():
    logger = logging.getLogger("phylo_correction.lg_paper")
    logger.setLevel(logging.DEBUG)
    fmt_str = "[%(asctime)s] - %(name)s - %(levelname)s - %(message)s"
    formatter = logging.Formatter(fmt_str)

    consoleHandler = logging.StreamHandler(sys.stdout)
    consoleHandler.setFormatter(formatter)
    logger.addHandler(consoleHandler)


init_logger()
logger = logging.getLogger("phylo_correction.lg_paper")


def verify_integrity(filepath: str, mode: str = "555"):
    if not os.path.exists(filepath):
        logger.error(
            f"Trying to verify the integrity of an inexistent file: {filepath}"
        )
        raise Exception(
            f"Trying to verify the integrity of an inexistent file: {filepath}"
        )
    mask = oct(os.stat(filepath).st_mode)[-3:]
    if mask != mode:
        logger.error(
            f"filename {filepath} does not have status {mode}. Instead, it "
            f"has status: {mask}. It is most likely corrupted."
        )
        raise Exception(
            f"filename {filepath} does not have status {mode}. Instead, it "
            f"has status: {mask}. It is most likely corrupted."
        )


def verify_integrity_of_directory(
    dirpath: str, expected_number_of_files: int, mode: str = "555"
):
    """
    Makes sure that the directory has the expected number of files and that
    they are all write protected (or another specified mode).
    """
    dirpath = os.path.abspath(dirpath)
    if not os.path.exists(dirpath):
        logger.error(
            f"Trying to verify the integrity of an inexistent "
            f"directory: {dirpath}"
        )
        raise Exception(
            f"Trying to verify the integrity of an inexistent "
            f"diretory: {dirpath}"
        )
    filenames = sorted(list(os.listdir(dirpath)))
    if len(filenames) != expected_number_of_files:
        raise Exception(
            f"{dirpath} already exists but does not contain the "
            "expected_number_of_files."
            f"\nExpected: {expected_number_of_files}\nFound: {len(filenames)}"
        )
    for filename in filenames:
        filepath = os.path.join(dirpath, filename)
        verify_integrity(filepath=filepath, mode=mode)


def wget_tarred_data_and_chmod(
    url: str,
    destination_directory: str,
    expected_number_of_files: int,
    mode: str = "555",
) -> None:
    """
    Download tar data from a url if not already present.

    Gets tarred data from `url` into `destination_directory` and chmods the
    data to 555 (or the `mode` specified) so that it is write protected.
    `expected_number_of_files` is the expected number of files after untarring.
    If the data is already present (which is determined by seeing whether the
    expected_number_of_files match), then the data is not downloaded again.

    Args:
        url: The url of the tar data.
        destination_directory: Where to untar the data to.
        expected_number_of_files: The expected number of files after
            untarring.
        mode: What mode to change the files to.

    Raises:
        Exception if the expected_number_of_files are not found after untarring,
            or if the data fails to download, etc.
    """
    destination_directory = os.path.abspath(destination_directory)
    if (
        os.path.exists(destination_directory)
        and len(os.listdir(destination_directory)) > 0
    ):
        verify_integrity_of_directory(
            dirpath=destination_directory,
            expected_number_of_files=expected_number_of_files,
            mode=mode,
        )
        logger.info(
            f"{url} has already been downloaded successfully to "
            f"{destination_directory}. Not downloading again."
        )
        return
    logger.info(f"Downloading {url} into {destination_directory}")
    if not os.path.exists(destination_directory):
        os.makedirs(destination_directory)
    logger.info(f"pushd into {destination_directory}")
    with pushd(destination_directory):
        wget.download(url)
        logger.info(f"wget {url} into {destination_directory}")
        os.system("tar -xvzf *.tar.gz >/dev/null")
        logger.info("Untarring file ...")
        os.system("rm *.tar.gz")
        logger.info("Removing tar file ...")
    os.system(f"chmod -R {mode} {destination_directory}")
    verify_integrity_of_directory(
        dirpath=destination_directory,
        expected_number_of_files=expected_number_of_files,
        mode=mode,
    )
    logger.info("Success!")


def _convert_lg_data(
    lg_data_dir: str,
    destination_directory: str,
) -> None:
    """
    Convert the LG MSAs from the PHYLIP format to our format.

    Args:
        lg_training_data_dir: Where the MSAs in PHYLIP format are.
        destination_directory: Where to write the converted MSAs to.
    """
    logger.info("Converting LG Training data to our MSA training format ...")
    if not os.path.exists(destination_directory):
        os.makedirs(destination_directory)
    protein_family_names = sorted(list(os.listdir(lg_data_dir)))
    for protein_family_name in protein_family_names:
        with open(
            os.path.join(lg_data_dir, protein_family_name), "r"
        ) as file:
            res = ""
            lines = file.read().split("\n")
            n_seqs, n_sites = map(int, lines[0].split(" "))
            for i in range(n_seqs):
                line = lines[2 + i]
                try:
                    protein_name, protein_sequence = line.split()
                except Exception:
                    raise ValueError(
                        f"For protein family {protein_family_name} , could "
                        f"not split line: {line}"
                    )
                assert len(protein_sequence) == n_sites
                res += f">{protein_name}\n"
                res += f"{protein_sequence}\n"
            output_filename = os.path.join(
                destination_directory,
                protein_family_name.replace(".", "_") + ".txt",
            )
            with open(output_filename, "w") as outfile:
                outfile.write(res)
                outfile.flush()
            os.system(f"chmod 555 {output_filename}")


def get_lg_PfamTestingAlignments_data(
    destination_directory: str,
) -> None:
    """
    Download the lg_PfamTestingAlignments data

    The data is hosted at:
    http://www.atgc-montpellier.fr/models/index.php?model=lg

    Args:
        destination_directory: Where to download the data to.
    """
    url = "http://www.atgc-montpellier.fr/download/datasets/models"\
        "/lg_PfamTestingAlignments.tar.gz"
    if (
        os.path.exists(destination_directory)
        and len(os.listdir(destination_directory)) > 0
    ):
        verify_integrity_of_directory(
            dirpath=destination_directory,
            expected_number_of_files=500,
            mode="555",
        )
        logger.info(
            f"{url} has already been downloaded successfully "
            f"to {destination_directory}. Not downloading again."
        )
        return
    with tempfile.TemporaryDirectory() as destination_directory_unprocessed:
        wget_tarred_data_and_chmod(
            url=url,
            destination_directory=destination_directory_unprocessed,
            expected_number_of_files=500,
            mode="777",
        )
        _convert_lg_data(
            lg_data_dir=destination_directory_unprocessed,
            destination_directory=destination_directory,
        )
    verify_integrity_of_directory(
        dirpath=destination_directory,
        expected_number_of_files=500,
        mode="555",
    )


def get_lg_PfamTrainingAlignments_data(
    destination_directory: str,
) -> None:
    """
    Get the lg_PfamTrainingAlignments.

    Downloads the lg_PfamTrainingAlignments data to the specified
    `destination_directory`, *converting it to our training MSA format in the
    process*.

    The data is hosted at:
    http://www.atgc-montpellier.fr/models/index.php?model=lg

    Args:
        destination_directory: Where to store the (converted) MSAs.
    """
    url = (
        "http://www.atgc-montpellier.fr/download/datasets/models"
        "/lg_PfamTrainingAlignments.tar.gz"
    )
    if (
        os.path.exists(destination_directory)
        and len(os.listdir(destination_directory)) > 0
    ):
        verify_integrity_of_directory(
            dirpath=destination_directory,
            expected_number_of_files=3912,
            mode="555",
        )
        logger.info(
            f"{url} has already been downloaded successfully "
            f"to {destination_directory}. Not downloading again."
        )
        return
    with tempfile.TemporaryDirectory() as destination_directory_unprocessed:
        wget_tarred_data_and_chmod(
            url=url,
            destination_directory=destination_directory_unprocessed,
            expected_number_of_files=1,
            mode="777",
        )
        _convert_lg_data(
            lg_data_dir=os.path.join(
                destination_directory_unprocessed, "AllData"
            ),
            destination_directory=destination_directory,
        )
    verify_integrity_of_directory(
        dirpath=destination_directory,
        expected_number_of_files=3912,
        mode="555",
    )


def get_rate_matrix_path_by_name(
    rate_matrix_name: str
) -> str:
    """
    Given a rate matrix name, returns the path to the rate matrix
    """
    if rate_matrix_name == "JTT":
        res = os.path.join(dir_path, "../../data/rate_matrices/lg.txt")
    else:
        raise ValueError(f"Unknown rate matrix name: {rate_matrix_name}")
    return res


MSADirType = str
FamiliesType = List[str]
RateMatrixPathType = str
PhylogenyEstimatorReturnType = Dict[str, str]
PhylogenyEstimatorType = Callable[
    [
        MSADirType,
        FamiliesType,
        RateMatrixPathType,
    ],
    PhylogenyEstimatorReturnType
]


def reproduce_lg_paper_fig_4(
    msa_dir_test: str,
    families_test: List[str],
    rate_matrix_names: List[str],
    baseline_rate_matrix: str,
    evaluation_phylogeny_estimator: PhylogenyEstimatorType,
    num_processed: int,
):
    """
    Reproduce Fig. 4 of the LG paper, adding the desired rate matrices.
    """
    for rate_matrix_name in rate_matrix_names:
        rate_matrix_path = get_rate_matrix_path_by_name(rate_matrix_name)
        output_likelihood_dir = evaluation_phylogeny_estimator(
            msa_dir=msa_dir_test,
            families=families_test,
            rate_matrix_path=rate_matrix_path,
        )["output_likelihood_dir"]
        print(output_likelihood_dir)