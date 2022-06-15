"""
Module to reproduce and extend all figures.

trRosetta dataset: https://www.pnas.org/doi/10.1073/pnas.1914677117

Prerequisites:
- input_data/a3m should point to the trRosetta alignments (e.g. via a symbolic
    link)
- input_data/pdb should point to the trRosetta structures (e.g. via a symbolic
    link)

The caching directories which contain all subsequent data are _cache_benchmarking
and _cache_lg_paper. You can similarly use a symbolic link to point to these.
"""
import os
from functools import partial
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from src.markov_chain import matrix_exponential
from src.io import write_count_matrices
from src.estimation import quantized_transitions_mle
from matplotlib.colors import LogNorm
from src.estimation import jtt_ipw

from src import caching, cherry_estimator, cherry_estimator_coevolution
from src.benchmarking.pfam_15k import (
    get_families_within_cutoff,
    simulate_ground_truth_data_coevolution,
    simulate_ground_truth_data_single_site,
)
from src.evaluation import (
    l_infty_norm,
    mean_relative_error,
    mre,
    plot_rate_matrix_predictions,
    relative_errors,
    rmse,
)
from src.io import read_count_matrices, read_mask_matrix, read_rate_matrix
from src.markov_chain import (
    get_equ_path,
    get_lg_path,
    get_lg_x_lg_path,
    normalized,
)
from src.phylogeny_estimation import gt_tree_estimator

PFAM_15K_MSA_DIR = "input_data/a3m"
PFAM_15K_PDB_DIR = "input_data/pdb"


def add_annotations_to_violinplot(
    yss_relative_errors: List[float],
    title: str,
):
    yticks = [np.log(10**i) for i in range(-5, 2)]
    ytickslabels = [f"$10^{{{i + 2}}}$" for i in range(-5, 2)]
    plt.grid()
    plt.ylabel("relative error (%)")
    plt.yticks(yticks, ytickslabels)
    for i, ys in enumerate(yss_relative_errors):
        ys = np.array(ys)
        label = "{:.1f}%".format(100 * np.median(ys))
        plt.annotate(
            label,  # this is the text
            (
                i + 0.05,
                np.log(np.max(ys)) - 1.5,
            ),  # these are the coordinates to position the label
            textcoords="offset points",  # how to position the text
            xytext=(0, 10),  # distance from text to points (x,y)
            ha="left",
            va="top",
            color="black",
            fontsize=12,
        )  # horizontal alignment can be left, right or center
        label = "{:.0f}%".format(100 * np.max(ys))
        plt.annotate(
            label,  # this is the text
            (
                i + 0.05,
                np.log(np.max(ys)),
            ),  # these are the coordinates to position the label
            textcoords="offset points",  # how to position the text
            xytext=(0, 10),  # distance from text to points (x,y)
            ha="left",
            va="top",
            color="red",
            fontsize=12,
        )  # horizontal alignment can be left, right or center
    plt.title(title + "\n(max and median error also reported)")
    plt.tight_layout()


def fig_pair_site_quantization_error():
    output_image_dir = "images/fig_pair_site_quantization_error"
    if not os.path.exists(output_image_dir):
        os.makedirs(output_image_dir)

    num_processes = 32
    num_sequences = (
        1024
    )
    num_rate_categories = (
        20
    )

    num_families_train = 15051
    num_families_test = 0

    quantization_grid_center = None
    quantization_grid_step = None
    quantization_grid_num_steps = None
    random_seed = 0
    learning_rate = 3e-2
    do_adam = True
    use_cpp_implementation = (
        True
    )
    minimum_distance_for_nontrivial_contact = (
        7
    )
    num_epochs = 200
    angstrom_cutoff = 8.0

    caching.set_cache_dir("_cache_benchmarking")
    caching.set_hash_len(64)

    qs = [
        (0.06, 445.79, 1),
        (0.06, 21.11, 2),
        (0.06, 4.59, 4),
        (0.06, 2.14, 8),
        (0.06, 1.46, 16),
        (0.06, 1.21, 32),
        #     (0.06, 1.1, 50),
        (0.06, 1.1, 64),
        (0.06, 1.048, 128),
        (0.06, 1.024, 256),
        #     (0.06, 1.012, 512),
        #     (0.06, 1.006, 1024),
    ]
    q_errors = [(np.sqrt(q[1]) - 1) * 100 for q in qs]
    q_points = [2 * q[2] + 1 for q in qs]
    yss_relative_errors = []
    Qs = []
    for (
        i,
        (
            quantization_grid_center,
            quantization_grid_step,
            quantization_grid_num_steps,
        ),
    ) in enumerate(qs):
        msg = f"***** grid = {(quantization_grid_center, quantization_grid_step, quantization_grid_num_steps)} *****"
        print("*" * len(msg))
        print(msg)
        print("*" * len(msg))

        families_all = get_families_within_cutoff(
            pfam_15k_msa_dir=PFAM_15K_MSA_DIR,
            min_num_sites=190 if num_families_train <= 1024 else 0,
            max_num_sites=230 if num_families_train <= 1024 else 1000000,
            min_num_sequences=1024 if num_families_train <= 1024 else 0,
            max_num_sequences=1000000,
        )
        families_train = families_all[:num_families_train]
        if num_families_test == 0:
            families_test = []
        else:
            families_test = families_all[-num_families_test:]
        print(f"len(families_all) = {len(families_all)}")
        if num_families_train + num_families_test > len(families_all):
            raise Exception(f"Training and testing set would overlap!")
        assert len(set(families_train + families_test)) == len(
            families_train
        ) + len(families_test)

        (
            msa_dir,
            contact_map_dir,
            gt_msa_dir,
            gt_tree_dir,
            gt_site_rates_dir,
            gt_likelihood_dir,
        ) = simulate_ground_truth_data_coevolution(
            pfam_15k_msa_dir=PFAM_15K_MSA_DIR,
            pfam_15k_pdb_dir=PFAM_15K_PDB_DIR,
            minimum_distance_for_nontrivial_contact=minimum_distance_for_nontrivial_contact,
            angstrom_cutoff=angstrom_cutoff,
            num_sequences=num_sequences,
            families=families_all,
            num_rate_categories=num_rate_categories,
            num_processes=num_processes,
            random_seed=random_seed,
            use_cpp_simulation_implementation=use_cpp_implementation,
        )

        cherry_estimator_res = cherry_estimator_coevolution(
            msa_dir=msa_dir,
            contact_map_dir=contact_map_dir,
            minimum_distance_for_nontrivial_contact=minimum_distance_for_nontrivial_contact,
            coevolution_mask_path="data/mask_matrices/aa_coevolution_mask.txt",
            families=families_train,
            tree_estimator=partial(
                gt_tree_estimator,
                gt_tree_dir=gt_tree_dir,
                gt_site_rates_dir=gt_site_rates_dir,
                gt_likelihood_dir=gt_likelihood_dir,
                num_rate_categories=num_rate_categories,
            ),
            initial_tree_estimator_rate_matrix_path=get_equ_path(),
            num_processes=num_processes,
            quantization_grid_center=quantization_grid_center,
            quantization_grid_step=quantization_grid_step,
            quantization_grid_num_steps=quantization_grid_num_steps,
            learning_rate=learning_rate,
            num_epochs=num_epochs,
            do_adam=do_adam,
            use_cpp_counting_implementation=use_cpp_implementation,
            num_processes_optimization=2,
        )

        print(
            f"tree_estimator_output_dirs_{i} = ",
            cherry_estimator_res["tree_estimator_output_dirs_0"],
        )

        count_matrices_dir = cherry_estimator_res["count_matrices_dir_0"]
        print(f"count_matrices_dir_{i} = {count_matrices_dir}")

        count_matrices = read_count_matrices(
            os.path.join(count_matrices_dir, "result.txt")
        )
        quantization_points = [
            float(x) for x in cherry_estimator_res["quantization_points"]
        ]
        plt.title("Number of transitions per time bucket")
        plt.bar(
            np.log(quantization_points),
            [x.to_numpy().sum().sum() for (_, x) in count_matrices],
        )
        plt.xlabel("Quantization Point")
        plt.ylabel("Number of Transitions")
        ticks = [0.0006, 0.006, 0.06, 0.6, 6.0]
        plt.xticks(np.log(ticks), ticks)
        plt.savefig(f"{output_image_dir}/count_matrices_{i}", dpi=300)
        plt.close()

        learned_rate_matrix_path = cherry_estimator_res[
            "learned_rate_matrix_path"
        ]
        print(f"learned_rate_matrix_path = {learned_rate_matrix_path}")

        learned_rate_matrix = read_rate_matrix(learned_rate_matrix_path)

        learned_rate_matrix = learned_rate_matrix.to_numpy()
        Qs.append(learned_rate_matrix)

        lg_x_lg = read_rate_matrix(
            get_lg_x_lg_path()
        ).to_numpy()
        mask_matrix = read_mask_matrix(
            "data/mask_matrices/aa_coevolution_mask.txt"
        ).to_numpy()

        yss_relative_errors.append(
            relative_errors(
                lg_x_lg,
                learned_rate_matrix,
                mask_matrix,
            )
        )

    for i in range(len(q_points)):
        plot_rate_matrix_predictions(lg_x_lg, Qs[i], mask_matrix)
        plt.title(
            f"True vs predicted rate matrix entries\nmax quantization error = %.1f%% (%i quantization points)"
            % (q_errors[i], q_points[i])
        )
        plt.tight_layout()
        plt.savefig(f"{output_image_dir}/log_log_plot_{i}", dpi=300)
        plt.close()

    df = pd.DataFrame(
        {
            "quantization points": sum(
                [
                    [q_points[i]] * len(yss_relative_errors[i])
                    for i in range(len(yss_relative_errors))
                ],
                [],
            ),
            "relative error": sum(yss_relative_errors, []),
        }
    )
    df["log relative error"] = np.log(df["relative error"])

    sns.violinplot(
        x="quantization points",
        y="log relative error",
        #     hue=None,
        data=df,
        #     palette="muted",
        inner=None,
        #     cut=0,
        #     bw=0.25
    )
    add_annotations_to_violinplot(
        yss_relative_errors,
        title="Distribution of relative error as quantization improves",
    )

    plt.savefig(f"{output_image_dir}/violin_plot", dpi=300)
    plt.close()


def fig_pair_site_number_of_families():
    output_image_dir = "images/fig_pair_site_number_of_families"
    if not os.path.exists(output_image_dir):
        os.makedirs(output_image_dir)

    num_processes = 32
    num_sequences = (
        1024
    )
    num_rate_categories = (
        20
    )

    num_families_train = None
    num_families_test = 0

    quantization_grid_center = 0.06
    quantization_grid_step = 1.1
    quantization_grid_num_steps = 50
    random_seed = 0
    learning_rate = 3e-2
    do_adam = True
    use_cpp_implementation = (
        True
    )
    minimum_distance_for_nontrivial_contact = (
        7
    )
    num_epochs = 200
    angstrom_cutoff = 8.0

    caching.set_cache_dir("_cache_benchmarking")
    caching.set_hash_len(64)

    num_families_train_list = [
        1,
        2,
        4,
        8,
        16,
        32,
        64,
        128,
        256,
        512,
        1024,
        2048,
        4096,
        8192,
        15051,
    ]
    yss_relative_errors = []
    Qs = []
    for (i, num_families_train) in enumerate(num_families_train_list):
        msg = f"***** num_families_train = {num_families_train} *****"
        print("*" * len(msg))
        print(msg)
        print("*" * len(msg))

        if num_families_train <= 1024:
            families_all = get_families_within_cutoff(
                pfam_15k_msa_dir=PFAM_15K_MSA_DIR,
                min_num_sites=190,
                max_num_sites=230,
                min_num_sequences=num_sequences,
                max_num_sequences=1000000,
            )
        else:
            families_all = get_families_within_cutoff(
                pfam_15k_msa_dir=PFAM_15K_MSA_DIR,
                min_num_sites=0,
                max_num_sites=1000000,
                min_num_sequences=0,
                max_num_sequences=1000000,
            )
        families_train = families_all[:num_families_train]
        if num_families_test == 0:
            families_test = []
        else:
            families_test = families_all[-num_families_test:]
        print(f"len(families_all) = {len(families_all)}")
        if num_families_train + num_families_test > len(families_all):
            raise Exception(f"Training and testing set would overlap!")
        assert len(set(families_train + families_test)) == len(
            families_train
        ) + len(families_test)

        (
            msa_dir,
            contact_map_dir,
            gt_msa_dir,
            gt_tree_dir,
            gt_site_rates_dir,
            gt_likelihood_dir,
        ) = simulate_ground_truth_data_coevolution(
            pfam_15k_msa_dir=PFAM_15K_MSA_DIR,
            pfam_15k_pdb_dir=PFAM_15K_PDB_DIR,
            minimum_distance_for_nontrivial_contact=minimum_distance_for_nontrivial_contact,
            angstrom_cutoff=angstrom_cutoff,
            num_sequences=num_sequences,
            families=families_all,
            num_rate_categories=num_rate_categories,
            num_processes=num_processes,
            random_seed=random_seed,
            use_cpp_simulation_implementation=use_cpp_implementation,
        )

        cherry_estimator_res = cherry_estimator_coevolution(
            msa_dir=msa_dir,
            contact_map_dir=contact_map_dir,
            minimum_distance_for_nontrivial_contact=minimum_distance_for_nontrivial_contact,
            coevolution_mask_path="data/mask_matrices/aa_coevolution_mask.txt",
            families=families_train,
            tree_estimator=partial(
                gt_tree_estimator,
                gt_tree_dir=gt_tree_dir,
                gt_site_rates_dir=gt_site_rates_dir,
                gt_likelihood_dir=gt_likelihood_dir,
                num_rate_categories=num_rate_categories,
            ),
            initial_tree_estimator_rate_matrix_path=get_equ_path(),
            num_processes=num_processes,
            quantization_grid_center=quantization_grid_center,
            quantization_grid_step=quantization_grid_step,
            quantization_grid_num_steps=quantization_grid_num_steps,
            learning_rate=learning_rate,
            num_epochs=num_epochs,
            do_adam=do_adam,
            use_cpp_counting_implementation=use_cpp_implementation,
            num_processes_optimization=2,
        )

        print(
            f"tree_estimator_output_dirs_{i} = ",
            cherry_estimator_res["tree_estimator_output_dirs_0"],
        )

        count_matrices_dir = cherry_estimator_res["count_matrices_dir_0"]
        print(f"count_matrices_dir_{i} = {count_matrices_dir}")
        # assert(False)
        count_matrices = read_count_matrices(
            os.path.join(count_matrices_dir, "result.txt")
        )
        quantization_points = [
            float(x) for x in cherry_estimator_res["quantization_points"]
        ]
        plt.title("Number of transitions per time bucket")
        plt.bar(
            np.log(quantization_points),
            [x.to_numpy().sum().sum() for (_, x) in count_matrices],
        )
        plt.xlabel("Quantization Point")
        plt.ylabel("Number of Transitions")
        ticks = [0.0006, 0.006, 0.06, 0.6, 6.0]
        plt.xticks(np.log(ticks), ticks)
        plt.savefig(f"{output_image_dir}/count_matrices_{i}", dpi=300)
        plt.close()

        learned_rate_matrix_path = cherry_estimator_res[
            "learned_rate_matrix_path"
        ]
        print(f"learned_rate_matrix_path = {learned_rate_matrix_path}")

        learned_rate_matrix = read_rate_matrix(learned_rate_matrix_path)

        learned_rate_matrix = learned_rate_matrix.to_numpy()
        Qs.append(learned_rate_matrix)

        lg_x_lg = read_rate_matrix(
            get_lg_x_lg_path()
        ).to_numpy()
        mask_matrix = read_mask_matrix(
            "data/mask_matrices/aa_coevolution_mask.txt"
        ).to_numpy()

        yss_relative_errors.append(
            relative_errors(
                lg_x_lg,
                learned_rate_matrix,
                mask_matrix,
            )
        )

    for i in range(len(num_families_train_list)):
        plot_rate_matrix_predictions(lg_x_lg, Qs[i], mask_matrix)
        plt.title(
            f"True vs predicted rate matrix entries\nnumber of families = %i"
            % num_families_train_list[i]
        )
        plt.tight_layout()
        plt.savefig(f"{output_image_dir}/log_log_plot_{i}", dpi=300)
        plt.close()

    df = pd.DataFrame(
        {
            "number of families": sum(
                [
                    [num_families_train_list[i]] * len(yss_relative_errors[i])
                    for i in range(len(yss_relative_errors))
                ],
                [],
            ),
            "relative error": sum(yss_relative_errors, []),
        }
    )
    df["log relative error"] = np.log(df["relative error"])

    sns.violinplot(
        x="number of families",
        y="log relative error",
        #     hue=None,
        data=df,
        #     palette="muted",
        inner=None,
        #     cut=0,
        #     bw=0.25
    )
    add_annotations_to_violinplot(
        yss_relative_errors,
        title="Distribution of relative error as sample size increases",
    )
    plt.savefig(f"{output_image_dir}/violin_plot", dpi=300)
    plt.close()
    print("Done!")


def live_demo_pair_of_sites():
    from functools import partial

    from src import caching, cherry_estimator_coevolution
    from src.benchmarking.pfam_15k import (
        compute_contact_maps,
        get_families,
        subsample_pfam_15k_msas,
    )
    from src.io import read_rate_matrix
    from src.markov_chain import get_lg_path, get_lg_x_lg_path
    from src.phylogeny_estimation import fast_tree

    PFAM_15K_MSA_DIR = "input_data/a3m"
    PFAM_15K_PDB_DIR = (
        "input_data/pdb"  # We'll need this for determining contacting residues!
    )

    caching.set_cache_dir("_cache_benchmarking")
    caching.set_hash_len(64)

    families = get_families(PFAM_15K_MSA_DIR)

    # Subsample the MSAs
    msa_dir = subsample_pfam_15k_msas(
        pfam_15k_msa_dir=PFAM_15K_MSA_DIR,
        num_sequences=1024,
        families=families,
        num_processes=32,
    )["output_msa_dir"]

    # Create contact maps
    contact_map_dir = compute_contact_maps(
        pfam_15k_pdb_dir=PFAM_15K_PDB_DIR,
        families=families,
        angstrom_cutoff=8.0,
        num_processes=32,
    )["output_contact_map_dir"]

    # Run the cherry method using FastTree tree estimator
    learned_rate_matrix_path = cherry_estimator_coevolution(
        msa_dir=msa_dir,
        contact_map_dir=contact_map_dir,
        minimum_distance_for_nontrivial_contact=7,
        coevolution_mask_path="data/mask_matrices/aa_coevolution_mask.txt",
        families=families,
        tree_estimator=partial(
            fast_tree,
            num_rate_categories=20,
        ),
        initial_tree_estimator_rate_matrix_path=get_lg_path(),
        num_processes=32,
        quantization_grid_center=0.06,
        quantization_grid_step=1.1,
        quantization_grid_num_steps=50,
        learning_rate=3e-2,
        num_epochs=200,
        do_adam=True,
        use_cpp_counting_implementation=True,
        num_processes_optimization=2,
    )["learned_rate_matrix_path"]
    learned_rate_matrix = read_rate_matrix(learned_rate_matrix_path).to_numpy()

    lg_x_lg = read_rate_matrix(get_lg_x_lg_path()).to_numpy()

    # Now compare matrices
    print("LGxLG matrix:")
    print(lg_x_lg[:3, :3])
    print("Learned rate matrix:")
    print(learned_rate_matrix[:3, :3])


def fig_lg_paper():
    """
    LG paper figure 4.
    """
    from src.benchmarking.lg_paper import reproduce_lg_paper_fig_4
    from src.phylogeny_estimation import phyml, fast_tree
    from typing import List
    from src.benchmarking.lg_paper import get_lg_PfamTrainingAlignments_data, get_lg_PfamTestingAlignments_data
    from src import caching
    from functools import partial
    import os

    num_processes = 4

    caching.set_cache_dir("_cache_lg_paper")
    caching.set_hash_len(64)

    LG_PFAM_TRAINING_ALIGNMENTS_DIR = "./lg_paper_data/lg_PfamTrainingAlignments"
    LG_PFAM_TESTING_ALIGNMENTS_DIR = "./lg_paper_data/lg_PfamTestingAlignments"

    get_lg_PfamTrainingAlignments_data(LG_PFAM_TRAINING_ALIGNMENTS_DIR)
    get_lg_PfamTestingAlignments_data(LG_PFAM_TESTING_ALIGNMENTS_DIR)

    output_image_dir = "images/lg_paper/"
    if not os.path.exists(output_image_dir):
        os.makedirs(output_image_dir)

    def get_families(
        msa_dir: str,
    ) -> List[str]:
        """
        TODO: Remove this function; import it from src.utils directly
        Get the list of protein families names.

        Args:
            msa_dir: Directory with the MSA files. There should be one file with
                name family.txt for each protein family.

        Returns:
            The list of protein family names in the provided directory.
        """
        families = sorted(list(os.listdir(msa_dir)))
        families = [x.split(".")[0] for x in families if x.endswith(".txt")]
        return families

    fast_tree_partial = partial(
        fast_tree,
        num_rate_categories=4,
        num_processes=num_processes,
        extra_command_line_args="-gamma",
    )

    phyml_partial = partial(
        phyml,
        num_rate_categories=4,
        num_processes=num_processes,
    )

    y, df, bootstraps, Qs = reproduce_lg_paper_fig_4(
        msa_train_dir=LG_PFAM_TRAINING_ALIGNMENTS_DIR,
        families_train=get_families(LG_PFAM_TRAINING_ALIGNMENTS_DIR),
        msa_test_dir=LG_PFAM_TESTING_ALIGNMENTS_DIR,
        families_test=get_families(LG_PFAM_TESTING_ALIGNMENTS_DIR),
        rate_estimator_names=[
            # "EQU",
            ("reported JTT", "JTT\n(reported)"),
            ("reproduced JTT", "JTT\n(reproduced)"),
            ("reported WAG", "WAG\n(reported)"),
            ("reproduced WAG", "WAG\n(reproduced)"),
            ("reported LG", "LG\n(reported)"),
            ("reproduced LG", "LG\n(reproduced)"),
            ("Cherry__1__3e-2__10000", "Cherry\n1st iteration"),
            ("Cherry__2__3e-2__10000", "Cherry\n2nd iteration"),
            ("Cherry__3__3e-2__10000", "Cherry\n3rd iteration"),
            ("Cherry__4__3e-2__10000", "Cherry\n4th iteration"),
            ("Cherry__5__3e-2__10000", "Cherry\n5th iteration"),
            ("Cherry__6__3e-2__10000", "Cherry\n6th iteration"),
        ],
        baseline_rate_estimator_name="reported JTT",
        evaluation_phylogeny_estimator=phyml_partial,
        # evaluation_phylogeny_estimator=fast_tree_partial,
        num_processes=num_processes,
        pfam_or_treebase='pfam',
        family_name_len=7,
        figsize=(14.4, 4.8),
        num_bootstraps=100,
        output_image_dir=output_image_dir,
    )


def fig_single_site_learning_rate_robustness():
    output_image_dir = "images/fig_single_site_learning_rate_robustness"
    if not os.path.exists(output_image_dir):
        os.makedirs(output_image_dir)

    num_processes = 1
    num_sequences = (
        1024
    )
    num_rate_categories = (
        20
    )

    # for num_families_train in [1024, 2048, 512, 256, 128, 4096, 8192, 15051, 64, 32, 16, 8, 4, 2, 1]:
    # for num_families_train in [1024, 2048, 512, 256, 128, 4096, 8192, 64, 32, 16, 8, 4, 2, 1]:
    # for num_families_train in [1024, 2048, 4096, 8192, 15051]:
    for num_families_train in [15051]:
        msg = f"***** num_families_train = {num_families_train} *****"
        print(len(msg) * "*")
        print(msg)
        print(len(msg) * "*")
        num_families_test = 0
        
        min_num_sites = 190
        max_num_sites = 230
        min_num_sequences = num_sequences
        max_num_sequences = 1000000
        rule_cutoff = 1024

        quantization_grid_center = 0.06
        quantization_grid_step = 1.1
        quantization_grid_num_steps = 50
        random_seed = 0
        learning_rate = None
        num_epochs = 10000
        do_adam = True
        use_cpp_implementation = (
            True
        )

        caching.set_cache_dir("_cache_benchmarking")
        caching.set_hash_len(64)

        learning_rates = [
        #     # 1e-6,
        #     # 3e-6,
        #     # 1e-5,
        #     # 3e-5,
        #     1e-4,
        #     3e-4,
        #     1e-3,
        #     3e-3,
        #     1e-2,
            3e-2,
            1e-1,
            3e-1,
        #     1e-0,
        #     # 3e-0,
        #     # 1e1,
        #     # 3e1,
        #     # 1e2,
        #     # 3e2,
        #     # 1e3,
        #     # 3e3,
        ]
        ys_mre = []
        yss_relative_errors = []
        Qs = []
        for i, lr in enumerate(learning_rates):
            msg = f"***** lr = {lr} *****"
            print("*" * len(msg))
            print(msg)
            print("*" * len(msg))

            families_all = get_families_within_cutoff(
                pfam_15k_msa_dir=PFAM_15K_MSA_DIR,
                min_num_sites=min_num_sites if num_families_train <= rule_cutoff else 0,
                max_num_sites=max_num_sites if num_families_train <= rule_cutoff else 1000000,
                min_num_sequences=min_num_sequences if num_families_train <= rule_cutoff else 0,
                max_num_sequences=max_num_sequences,
            )
            families_train = families_all[:num_families_train]
            if num_families_test == 0:
                families_test = []
            else:
                families_test = families_all[-num_families_test:]
            print(f"len(families_all) = {len(families_all)}")
            if num_families_train + num_families_test > len(families_all):
                raise Exception(f"Training and testing set would overlap!")
            assert len(set(families_train + families_test)) == len(
                families_train
            ) + len(families_test)

            (
                msa_dir,
                contact_map_dir,
                gt_msa_dir,
                gt_tree_dir,
                gt_site_rates_dir,
                gt_likelihood_dir,
            ) = simulate_ground_truth_data_single_site(
                pfam_15k_msa_dir=PFAM_15K_MSA_DIR,
                num_sequences=num_sequences,
                families=families_train + families_test,
                num_rate_categories=num_rate_categories,
                num_processes=num_processes,
                random_seed=random_seed,
                use_cpp_simulation_implementation=use_cpp_implementation,
            )

            cherry_estimator_res = cherry_estimator(
                msa_dir=msa_dir,
                families=families_train,
                tree_estimator=partial(
                    gt_tree_estimator,
                    gt_tree_dir=gt_tree_dir,
                    gt_site_rates_dir=gt_site_rates_dir,
                    gt_likelihood_dir=gt_likelihood_dir,
                    num_rate_categories=num_rate_categories,
                ),
                initial_tree_estimator_rate_matrix_path=get_equ_path(),
                num_iterations=1,
                num_processes=num_processes,
                quantization_grid_center=quantization_grid_center,
                quantization_grid_step=quantization_grid_step,
                quantization_grid_num_steps=quantization_grid_num_steps,
                learning_rate=lr,
                num_epochs=num_epochs,
                do_adam=do_adam,
                use_cpp_counting_implementation=use_cpp_implementation,
                num_processes_optimization=2,
            )

            try:

                print(
                    f"tree_estimator_output_dirs_{i} = ",
                    cherry_estimator_res["tree_estimator_output_dirs_0"],
                )

                count_matrices_dir = cherry_estimator_res["count_matrices_dir_0"]
                print(f"count_matrices_dir_{i} = {count_matrices_dir}")

                count_matrices = read_count_matrices(
                    os.path.join(count_matrices_dir, "result.txt")
                )
                quantization_points = [
                    float(x) for x in cherry_estimator_res["quantization_points"]
                ]
                plt.title("Number of transitions per time bucket")
                plt.bar(
                    np.log(quantization_points),
                    [x.to_numpy().sum().sum() for (_, x) in count_matrices],
                )
                plt.xlabel("Quantization Point")
                plt.ylabel("Number of Transitions")
                ticks = [0.0006, 0.006, 0.06, 0.6, 6.0]
                plt.xticks(np.log(ticks), ticks)
                plt.savefig(f"{output_image_dir}/count_matrices_{i}_{num_families_train}", dpi=300)
                plt.close()

                learned_rate_matrix_path = cherry_estimator_res[
                    "learned_rate_matrix_path"
                ]

                learned_rate_matrix = read_rate_matrix(learned_rate_matrix_path)

                learned_rate_matrix = learned_rate_matrix.to_numpy()
                Qs.append(learned_rate_matrix)

                lg = read_rate_matrix(get_lg_path()).to_numpy()

                yss_relative_errors.append(relative_errors(lg, learned_rate_matrix))

            except:
                pass

        for i in range(len(learning_rates)):
            try:
                plot_rate_matrix_predictions(
                    read_rate_matrix(get_lg_path()).to_numpy(), Qs[i]
                )
                plt.title(
                    f"True vs predicted rate matrix entries\nlearning rate = %f"
                    % lr
                )
                plt.tight_layout()
                plt.savefig(f"{output_image_dir}/log_log_plot_{i}_{num_families_train}", dpi=300)
                plt.close()
            except:
                pass

        try:
            df = pd.DataFrame(
                {
                    "learning rate": sum(
                        [
                            [learning_rates[i]] * len(yss_relative_errors[i])
                            for i in range(len(yss_relative_errors))
                        ],
                        [],
                    ),
                    "relative error": sum(yss_relative_errors, []),
                }
            )
            df["log relative error"] = np.log(df["relative error"])

            sns.violinplot(
                x="learning rate",
                y="log relative error",
                #     hue=None,
                data=df,
                #     palette="muted",
                inner=None,
                #     cut=0,
                #     bw=0.25
            )
            add_annotations_to_violinplot(
                yss_relative_errors,
                title="Distribution of relative error as learning rate varies",
            )
            plt.savefig(f"{output_image_dir}/violin_plot_{num_families_train}", dpi=300)
            plt.close()
        except:
            pass


def debug_pytorch_optimizer():
    """
    Test that the pytorch optimizer converges, and does better with more data.

    No caching used here since I am debugging.
    """
    from src.markov_chain import matrix_exponential
    from src.io import write_count_matrices
    from src.estimation import quantized_transitions_mle

    # Hyperparameters of the test
    samples_per_row = 100000000
    sample_repetitions = 1
    learning_rate = 3e-2
    do_adam = True
    num_epochs = 10000

    Q_df = read_rate_matrix(get_lg_path())
    Q_numpy = Q_df.to_numpy()

    quantization_grid_center = 0.06
    quantization_grid_step = 1.1
    quantization_grid_num_steps = 50
    # quantization_grid_num_steps = 0

    quantization_points_str = [
        ("%.5f" % (quantization_grid_center * quantization_grid_step**i))
        for i in range(
            -quantization_grid_num_steps, quantization_grid_num_steps + 1, 1
        )
    ]
    quantization_points = map(float, quantization_points_str)

    # Fist create synthetic data.
    count_matrices = [
        [
            q,
            sample_repetitions * pd.DataFrame(
                (
                    samples_per_row * matrix_exponential(
                        exponents=np.array([q]),
                        Q=Q_numpy,
                        fact=None,
                        reversible=False,
                        device='cpu',
                    ).reshape([Q_numpy.shape[0], Q_numpy.shape[1]])
                ).astype(int),
                columns=Q_df.columns,
                index=Q_df.index,
            ),
        ]
        for q in quantization_points
    ]
    count_matrices_path = "./count_matrices_path.txt"
    write_count_matrices(
        count_matrices=count_matrices,
        count_matrices_path=count_matrices_path,
    )

    # Then run the optimizer.
    initialization_path = get_equ_path()
    output_rate_matrix_dir = "output_rate_matrix_dir"
    os.system(f"chmod -R 777 {output_rate_matrix_dir}")
    quantized_transitions_mle(
        count_matrices_path=count_matrices_path,
        initialization_path=initialization_path,
        mask_path=None,
        output_rate_matrix_dir=output_rate_matrix_dir,
        stationary_distribution_path=None,
        rate_matrix_parameterization="pande_reversible",
        device="cpu",
        learning_rate=learning_rate,
        num_epochs=num_epochs,
        do_adam=do_adam,
        OMP_NUM_THREADS=2,
        OPENBLAS_NUM_THREADS=2,
    )

    learned_rate_matrix = read_rate_matrix(
        os.path.join(output_rate_matrix_dir, "result.txt")
    ).to_numpy()

    res = relative_errors(Q_numpy, learned_rate_matrix)
    print(f"mean relative error: {np.mean(res)}")
    print(f"max relative error: {np.max(res)}")


@caching.cached_computation(
    output_dirs=["output_count_matrices_dir"],
)
def create_synthetic_count_matrices(
    quantization_points: List[float],
    samples_per_row: int,
    rate_matrix_path: str,
    output_count_matrices_dir: Optional[str] = None,
):
    """
    Create synthetic count matrices.

    Args:
        quantization_points: The branch lengths.
        samples_per_row: For each branch length, this number of transitions will
            be observed for each state (up to transitions lost from taking
            floor)
        rate_matrix_path: Path to the ground truth rate matrix.
    """
    Q_df = read_rate_matrix(rate_matrix_path)
    Q_numpy = Q_df.to_numpy()
    count_matrices = [
        [
            q,
            pd.DataFrame(
                (
                    samples_per_row * matrix_exponential(
                        exponents=np.array([q]),
                        Q=Q_numpy,
                        fact=None,
                        reversible=False,
                        device='cpu',
                    ).reshape([Q_numpy.shape[0], Q_numpy.shape[1]])
                ).astype(int),
                columns=Q_df.columns,
                index=Q_df.index,
            ),
        ]
        for q in quantization_points
    ]
    write_count_matrices(
        count_matrices, os.path.join(output_count_matrices_dir, "result.txt")
    )


def fig_convergence_on_infinite_data_single_site():
    """
    We show that on "infinite" single-site data, the pytorch optimizer converges
    to the solution for a variety of learning rates, and we identify the optimal
    learning rate to be 0.1: small - so as to be numerically stable - but not
    too small - so as to converge fast.
    """
    caching.set_cache_dir("_cache_benchmarking")
    caching.set_hash_len(64)

    output_image_dir = "images/fig_convergence_on_infinite_data_single_site"
    if not os.path.exists(output_image_dir):
        print(f"Creating {output_image_dir}")
        os.makedirs(output_image_dir)

    # Hyperparameters of the Adam optimizer
    learning_rate_grid = [
        3e-3,
        1e-2,
        3e-2,
        1e-1,
        3e-1,
        1e-0,
        # 3e-0,  # Training diverges starting at this learning rate
    ]
    num_epochs_grid = [
        1,
        2,
        4,
        8,
        16,
        32,
        64,
        128,
        256,
        512,
        1024,
        2048,
        4096,
        8192,
        16384,
        32768,
    ]

    rate_matrix_path = get_lg_path()
    Q_df = read_rate_matrix(rate_matrix_path)
    Q_numpy = Q_df.to_numpy()

    # Create synthetic training data (in the form of count matrices)
    output_count_matrices_dir = create_synthetic_count_matrices(
        quantization_points=[0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
        samples_per_row=100000000,
        rate_matrix_path=rate_matrix_path,
    )["output_count_matrices_dir"]
    count_matrices_path = os.path.join(output_count_matrices_dir, "result.txt")

    result_tuples = []
    res_2d = {'mean': [], 'median': [], 'max': []}

    for learning_rate in learning_rate_grid:
        res_2d_row = {'mean': [], 'median': [], 'max': []}
        for num_epochs in num_epochs_grid:
            initialization_path = get_equ_path()
            # Run the Adam optimizer.
            output_rate_matrix_dir = quantized_transitions_mle(
                count_matrices_path=count_matrices_path,
                initialization_path=initialization_path,
                mask_path=None,
                stationary_distribution_path=None,
                rate_matrix_parameterization="pande_reversible",
                device="cpu",
                learning_rate=learning_rate,
                num_epochs=num_epochs,
                do_adam=True,
                OMP_NUM_THREADS=2,
                OPENBLAS_NUM_THREADS=2,
            )["output_rate_matrix_dir"]

            learned_rate_matrix = read_rate_matrix(
                os.path.join(output_rate_matrix_dir, "result.txt")
            ).to_numpy()

            res = relative_errors(Q_numpy, learned_rate_matrix)
            result_tuples.append((learning_rate, num_epochs, np.mean(res), np.median(res), np.max(res)))
            res_2d_row['mean'].append(np.mean(res))
            res_2d_row['median'].append(np.median(res))
            res_2d_row['max'].append(np.max(res))
        res_2d['mean'].append(res_2d_row['mean'])
        res_2d['median'].append(res_2d_row['median'])
        res_2d['max'].append(res_2d_row['max'])

    res = pd.DataFrame(
        result_tuples,
        columns=["learning_rate", "num_epochs", "mean_relative_error", "median_relative_error", "max_relative_error"]
    )
    # print(res)

    for metric_name in ['max', 'median']:
        sns.heatmap(
            np.array(res_2d[metric_name]).T,
            yticklabels=num_epochs_grid,
            xticklabels=learning_rate_grid,
            cmap="YlGnBu",  # "RdBu_r"
            annot=True,
            annot_kws={"size": 6},
            fmt=".1",
            # vmin=0,
            # vmax=vmax,
            # center=center,
            norm=LogNorm(),
        )
        plt.xlabel("learning rate")
        plt.ylabel("number of epochs")
        plt.title(f"{metric_name} relative error")
        # plt.gcf().set_size_inches(16, 16)
        plt.tight_layout()
        plt.savefig(f"{output_image_dir}/heatmap_{metric_name}.png", dpi=300)
        plt.close()


def fig_convergence_on_large_data_single_site():
    """
    We show that on single-site data simulated on top of real trees, the pytorch
    optimizer converges to the solution for a variety of learning rates, and we
    validate the optimal learning rate to be 0.1. This figure provides more
    information than:
    fig_convergence_on_infinite_data_single_site
    in that the branch lengths come from real data, and they are also getting
    quntized.
    """
    caching.set_cache_dir("_cache_benchmarking")
    caching.set_hash_len(64)

    output_image_dir = "images/fig_convergence_on_large_data_single_site"
    if not os.path.exists(output_image_dir):
        os.makedirs(output_image_dir)

    # Hyperparameters of the Adam optimizer
    learning_rate_grid = [
        3e-3,
        1e-2,
        3e-2,
        1e-1,
        3e-1,
        1e-0,
        # 3e-0,  # Training diverges starting at this learning rate
    ]
    num_epochs_grid = [
        1,
        2,
        4,
        8,
        16,
        32,
        64,
        128,
        256,
        512,
        1024,
        2048,
        4096,
        8192,
        16384,
        32768,
    ]

    Q_numpy = read_rate_matrix(get_lg_path()).to_numpy()

    num_processes = 2
    num_sequences = (
        1024
    )
    num_rate_categories = (
        20
    )

    result_tuples = []
    res_2d = {'mean': [], 'median': [], 'max': []}

    for learning_rate in learning_rate_grid:
        res_2d_row = {'mean': [], 'median': [], 'max': []}
        for num_epochs in num_epochs_grid:
            num_families_train = 15051
            num_families_test = 0
            
            quantization_grid_center = 0.03
            quantization_grid_step = 1.1
            quantization_grid_num_steps = 64
            random_seed = 0
            use_cpp_implementation = (
                True
            )

            families_all = get_families_within_cutoff(
                pfam_15k_msa_dir=PFAM_15K_MSA_DIR,
                min_num_sites=0,
                max_num_sites=1000000,
                min_num_sequences=0,
                max_num_sequences=1000000,
            )
            families_train = families_all[:num_families_train]
            if num_families_test == 0:
                families_test = []
            else:
                families_test = families_all[-num_families_test:]
            if num_families_train + num_families_test > len(families_all):
                raise Exception(f"Training and testing set would overlap!")
            assert len(set(families_train + families_test)) == len(
                families_train
            ) + len(families_test)

            (
                msa_dir,
                contact_map_dir,
                gt_msa_dir,
                gt_tree_dir,
                gt_site_rates_dir,
                gt_likelihood_dir,
            ) = simulate_ground_truth_data_single_site(
                pfam_15k_msa_dir=PFAM_15K_MSA_DIR,
                num_sequences=num_sequences,
                families=families_train + families_test,
                num_rate_categories=num_rate_categories,
                num_processes=num_processes,
                random_seed=random_seed,
                use_cpp_simulation_implementation=use_cpp_implementation,
            )

            cherry_estimator_res = cherry_estimator(
                msa_dir=msa_dir,
                families=families_train,
                tree_estimator=partial(
                    gt_tree_estimator,
                    gt_tree_dir=gt_tree_dir,
                    gt_site_rates_dir=gt_site_rates_dir,
                    gt_likelihood_dir=gt_likelihood_dir,
                    num_rate_categories=num_rate_categories,
                ),
                initial_tree_estimator_rate_matrix_path=get_equ_path(),
                num_iterations=1,
                num_processes=num_processes,
                quantization_grid_center=quantization_grid_center,
                quantization_grid_step=quantization_grid_step,
                quantization_grid_num_steps=quantization_grid_num_steps,
                learning_rate=learning_rate,
                num_epochs=num_epochs,
                do_adam=True,
                use_cpp_counting_implementation=use_cpp_implementation,
                num_processes_optimization=2,
            )

            learned_rate_matrix_path = cherry_estimator_res[
                "learned_rate_matrix_path"
            ]
            learned_rate_matrix = read_rate_matrix(learned_rate_matrix_path)
            learned_rate_matrix = learned_rate_matrix.to_numpy()
            res = relative_errors(Q_numpy, learned_rate_matrix)
            result_tuples.append((learning_rate, num_epochs, np.mean(res), np.median(res), np.max(res)))
            res_2d_row['mean'].append(np.mean(res))
            res_2d_row['median'].append(np.median(res))
            res_2d_row['max'].append(np.max(res))
        res_2d['mean'].append(res_2d_row['mean'])
        res_2d['median'].append(res_2d_row['median'])
        res_2d['max'].append(res_2d_row['max'])

    res = pd.DataFrame(
        result_tuples,
        columns=["learning_rate", "num_epochs", "mean_relative_error", "median_relative_error", "max_relative_error"]
    )
    # print(res)

    for metric_name in ['max', 'median']:
        sns.heatmap(
            np.array(res_2d[metric_name]).T,
            yticklabels=num_epochs_grid,
            xticklabels=learning_rate_grid,
            cmap="YlGnBu",  # "RdBu_r"
            annot=True,
            annot_kws={"size": 6},
            fmt=".1",
            # vmin=0,
            # vmax=vmax,
            # center=center,
            norm=LogNorm(),
        )
        plt.xlabel("learning rate")
        plt.ylabel("number of epochs")
        plt.title(f"{metric_name} relative error")
        # plt.gcf().set_size_inches(16, 16)
        plt.tight_layout()
        plt.savefig(f"{output_image_dir}/heatmap_{metric_name}.png", dpi=300)
        plt.close()


def fig_single_site_quantization_error():
    """
    We show that ~100 quantization points (geometric increments of 10%) is
    enough.
    """
    caching.set_cache_dir("_cache_benchmarking")
    caching.set_hash_len(64)

    output_image_dir = "images/fig_single_site_quantization_error"
    if not os.path.exists(output_image_dir):
        os.makedirs(output_image_dir)

    num_processes = 32
    num_sequences = (
        1024
    )
    num_rate_categories = (
        20
    )

    num_families_train = 15051
    num_families_test = 0

    quantization_grid_center = None
    quantization_grid_step = None
    quantization_grid_num_steps = None
    random_seed = 0
    learning_rate = 1e-1
    num_epochs = 2000
    use_cpp_implementation = (
        True
    )

    qs = [
        (0.03, 445.79, 1),
        (0.03, 21.11, 2),
        (0.03, 4.59, 4),
        (0.03, 2.14, 8),
        (0.03, 1.46, 16),
        (0.03, 1.21, 32),
        (0.03, 1.1, 64),
        (0.03, 1.048, 128),
        (0.03, 1.024, 256),
    ]
    q_errors = [(np.sqrt(q[1]) - 1) * 100 for q in qs]
    q_points = [2 * q[2] + 1 for q in qs]
    ys_mre = []
    yss_relative_errors = []
    Qs = []
    for (
        i,
        (
            quantization_grid_center,
            quantization_grid_step,
            quantization_grid_num_steps,
        ),
    ) in enumerate(qs):
        msg = f"***** grid = {(quantization_grid_center, quantization_grid_step, quantization_grid_num_steps)} *****"
        print("*" * len(msg))
        print(msg)
        print("*" * len(msg))

        families_all = get_families_within_cutoff(
            pfam_15k_msa_dir=PFAM_15K_MSA_DIR,
            min_num_sites=190 if num_families_train <= 1024 else 0,
            max_num_sites=230 if num_families_train <= 1024 else 1000000,
            min_num_sequences=1024 if num_families_train <= 1024 else 0,
            max_num_sequences=1000000,
        )
        families_train = families_all[:num_families_train]
        if num_families_test == 0:
            families_test = []
        else:
            families_test = families_all[-num_families_test:]
        print(f"len(families_all) = {len(families_all)}")
        if num_families_train + num_families_test > len(families_all):
            raise Exception(f"Training and testing set would overlap!")
        assert len(set(families_train + families_test)) == len(
            families_train
        ) + len(families_test)

        (
            msa_dir,
            contact_map_dir,
            gt_msa_dir,
            gt_tree_dir,
            gt_site_rates_dir,
            gt_likelihood_dir,
        ) = simulate_ground_truth_data_single_site(
            pfam_15k_msa_dir=PFAM_15K_MSA_DIR,
            num_sequences=num_sequences,
            families=families_train + families_test,
            num_rate_categories=num_rate_categories,
            num_processes=num_processes,
            random_seed=random_seed,
            use_cpp_simulation_implementation=use_cpp_implementation,
        )

        cherry_estimator_res = cherry_estimator(
            msa_dir=msa_dir,
            families=families_train,
            tree_estimator=partial(
                gt_tree_estimator,
                gt_tree_dir=gt_tree_dir,
                gt_site_rates_dir=gt_site_rates_dir,
                gt_likelihood_dir=gt_likelihood_dir,
                num_rate_categories=num_rate_categories,
            ),
            initial_tree_estimator_rate_matrix_path=get_equ_path(),
            num_iterations=1,
            num_processes=num_processes,
            quantization_grid_center=quantization_grid_center,
            quantization_grid_step=quantization_grid_step,
            quantization_grid_num_steps=quantization_grid_num_steps,
            learning_rate=learning_rate,
            num_epochs=num_epochs,
            do_adam=True,
            use_cpp_counting_implementation=use_cpp_implementation,
            num_processes_optimization=2,
        )

        print(
            f"tree_estimator_output_dirs_{i} = ",
            cherry_estimator_res["tree_estimator_output_dirs_0"],
        )

        count_matrices_dir = cherry_estimator_res["count_matrices_dir_0"]
        print(f"count_matrices_dir_{i} = {count_matrices_dir}")
        # assert(False)
        count_matrices = read_count_matrices(
            os.path.join(count_matrices_dir, "result.txt")
        )
        quantization_points = [
            float(x) for x in cherry_estimator_res["quantization_points"]
        ]
        plt.title("Number of transitions per time bucket")
        plt.bar(
            np.log(quantization_points),
            [x.to_numpy().sum().sum() for (_, x) in count_matrices],
        )
        plt.xlabel("Quantization Point")
        plt.ylabel("Number of Transitions")
        ticks = [0.0003, 0.003, 0.03, 0.3, 3.0]
        plt.xticks(np.log(ticks), ticks)
        plt.savefig(f"{output_image_dir}/count_matrices_{i}", dpi=300)
        plt.close()

        learned_rate_matrix_path = cherry_estimator_res[
            "learned_rate_matrix_path"
        ]

        learned_rate_matrix = read_rate_matrix(learned_rate_matrix_path)

        learned_rate_matrix = learned_rate_matrix.to_numpy()
        Qs.append(learned_rate_matrix)

        lg = read_rate_matrix(get_lg_path()).to_numpy()

        yss_relative_errors.append(relative_errors(lg, learned_rate_matrix))

    for i in range(len(q_points)):
        plot_rate_matrix_predictions(
            read_rate_matrix(get_lg_path()).to_numpy(), Qs[i]
        )
        plt.title(
            f"True vs predicted rate matrix entries\nmax quantization error = %.1f%% (%i quantization points)"
            % (q_errors[i], q_points[i])
        )
        plt.tight_layout()
        plt.savefig(f"{output_image_dir}/log_log_plot_{i}", dpi=300)
        plt.close()

    df = pd.DataFrame(
        {
            "quantization points": sum(
                [
                    [q_points[i]] * len(yss_relative_errors[i])
                    for i in range(len(yss_relative_errors))
                ],
                [],
            ),
            "relative error": sum(yss_relative_errors, []),
        }
    )
    df["log relative error"] = np.log(df["relative error"])

    sns.violinplot(
        x="quantization points",
        y="log relative error",
        #     hue=None,
        data=df,
        #     palette="muted",
        inner=None,
        #     cut=0,
        #     bw=0.25
    )
    add_annotations_to_violinplot(
        yss_relative_errors,
        title="Distribution of relative error as quantization improves",
    )
    plt.savefig(f"{output_image_dir}/violin_plot", dpi=300)
    plt.close()


def fig_single_site_cherry_vs_edge():
    """
    We compare the efficiency of our Cherry method ("cherry") against that of
    the oracle method ("edge"), and show that it is off by 4-8x, as suggested
    by the back-of-envelope estimate.
    """
    caching.set_cache_dir("_cache_benchmarking")
    caching.set_hash_len(64)

    for edge_or_cherry in ["edge", "cherry"]:
        output_image_dir = (
            f"images/fig_single_site_cherry_vs_edge/{edge_or_cherry}"
        )
        if not os.path.exists(output_image_dir):
            os.makedirs(output_image_dir)

        num_processes = 32
        num_sequences = 1024
        num_rate_categories = (
            20
        )

        num_families_train = None
        num_families_test = 0
        min_num_sites = 190
        max_num_sites = 230
        min_num_sequences = num_sequences
        max_num_sequences = 1000000

        quantization_grid_center = 0.03
        quantization_grid_step = 1.1
        quantization_grid_num_steps = 64
        random_seed = 0
        learning_rate = 1e-1
        num_epochs = 2000
        use_cpp_implementation = (
            True
        )

        # num_families_train_list = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
        num_families_train_list = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 15051]

        yss_relative_errors = []
        Qs = []
        for (i, num_families_train) in enumerate(num_families_train_list):
            msg = f"***** num_families_train = {num_families_train} *****"
            print("*" * len(msg))
            print(msg)
            print("*" * len(msg))

            families_all = get_families_within_cutoff(
                pfam_15k_msa_dir=PFAM_15K_MSA_DIR,
                min_num_sites=min_num_sites if num_families_train <= 1024 else 0,
                max_num_sites=max_num_sites if num_families_train <= 1024 else 1000000,
                min_num_sequences=min_num_sequences if num_families_train <= 1024 else 0,
                max_num_sequences=max_num_sequences,
            )
            families_train = families_all[:num_families_train]
            if num_families_test == 0:
                families_test = []
            else:
                families_test = families_all[-num_families_test:]
            print(f"len(families_all) = {len(families_all)}")
            if num_families_train + num_families_test > len(families_all):
                raise Exception(f"Training and testing set would overlap!")
            assert len(set(families_train + families_test)) == len(
                families_train
            ) + len(families_test)

            (
                msa_dir,
                contact_map_dir,
                gt_msa_dir,
                gt_tree_dir,
                gt_site_rates_dir,
                gt_likelihood_dir,
            ) = simulate_ground_truth_data_single_site(
                pfam_15k_msa_dir=PFAM_15K_MSA_DIR,
                num_sequences=num_sequences,
                families=families_all,
                num_rate_categories=num_rate_categories,
                num_processes=num_processes,
                random_seed=random_seed,
                use_cpp_simulation_implementation=use_cpp_implementation,
            )

            # Now run the cherry and oracle edge methods.
            print(f"**** edge_or_cherry = {edge_or_cherry} *****")
            cherry_estimator_res = cherry_estimator(
                msa_dir=msa_dir
                if edge_or_cherry == "cherry"
                else gt_msa_dir,
                families=families_train,
                tree_estimator=partial(
                    gt_tree_estimator,
                    gt_tree_dir=gt_tree_dir,
                    gt_site_rates_dir=gt_site_rates_dir,
                    gt_likelihood_dir=gt_likelihood_dir,
                    num_rate_categories=num_rate_categories,
                ),
                initial_tree_estimator_rate_matrix_path=get_equ_path(),
                num_iterations=1,
                num_processes=2,
                quantization_grid_center=quantization_grid_center,
                quantization_grid_step=quantization_grid_step,
                quantization_grid_num_steps=quantization_grid_num_steps,
                learning_rate=learning_rate,
                num_epochs=num_epochs,
                do_adam=True,
                edge_or_cherry=edge_or_cherry,
                use_cpp_counting_implementation=use_cpp_implementation,
                num_processes_optimization=2,
            )

            learned_rate_matrix_path = cherry_estimator_res[
                "learned_rate_matrix_path"
            ]
            learned_rate_matrix = read_rate_matrix(learned_rate_matrix_path)
            learned_rate_matrix = learned_rate_matrix.to_numpy()

            lg = read_rate_matrix(get_lg_path()).to_numpy()
            print(
                f"tree_estimator_output_dirs_{i} = ",
                cherry_estimator_res["tree_estimator_output_dirs_0"],
            )

            learned_rate_matrix_path = cherry_estimator_res[
                "learned_rate_matrix_path"
            ]
            print(f"learned_rate_matrix_path = {learned_rate_matrix_path}")
            learned_rate_matrix = read_rate_matrix(learned_rate_matrix_path)

            learned_rate_matrix = learned_rate_matrix.to_numpy()
            Qs.append(learned_rate_matrix)

            lg = read_rate_matrix(get_lg_path()).to_numpy()

            yss_relative_errors.append(relative_errors(lg, learned_rate_matrix))

        for i in range(len(num_families_train_list)):
            plot_rate_matrix_predictions(
                read_rate_matrix(get_lg_path()).to_numpy(), Qs[i]
            )
            plt.title(
                f"True vs predicted rate matrix entries\nnumber of families = %i"
                % num_families_train_list[i]
            )
            plt.tight_layout()
            plt.savefig(f"{output_image_dir}/log_log_plot_{i}", dpi=300)
            plt.close()

        df = pd.DataFrame(
            {
                "number of families": sum(
                    [
                        [num_families_train_list[i]]
                        * len(yss_relative_errors[i])
                        for i in range(len(yss_relative_errors))
                    ],
                    [],
                ),
                "relative error": sum(yss_relative_errors, []),
            }
        )
        df["log relative error"] = np.log(df["relative error"])

        sns.violinplot(
            x="number of families",
            y="log relative error",
            #     hue=None,
            data=df,
            #     palette="muted",
            inner=None,
            #     cut=0,
            #     bw=0.25
        )
        add_annotations_to_violinplot(
            yss_relative_errors,
            title="Distribution of relative error as sample size increases",
        )
        plt.savefig(f"{output_image_dir}/violin_plot", dpi=300)
        plt.close()


def live_demo_single_site():
    from functools import partial

    from src import caching, cherry_estimator
    from src.benchmarking.pfam_15k import get_families, subsample_pfam_15k_msas
    from src.io import read_rate_matrix
    from src.markov_chain import get_lg_path
    from src.phylogeny_estimation import fast_tree

    PFAM_15K_MSA_DIR = "input_data/a3m"

    caching.set_cache_dir("_cache_benchmarking")
    caching.set_hash_len(64)

    families = get_families(PFAM_15K_MSA_DIR)

    # Subsample the MSAs
    msa_dir_train = subsample_pfam_15k_msas(
        pfam_15k_msa_dir=PFAM_15K_MSA_DIR,
        num_sequences=1024,
        families=families,
        num_processes=32,
    )["output_msa_dir"]

    # Run the cherry method using FastTree tree estimator
    learned_rate_matrix_path = cherry_estimator(
        msa_dir=msa_dir_train,
        families=families,
        tree_estimator=partial(
            fast_tree,
            num_rate_categories=20,
        ),
        initial_tree_estimator_rate_matrix_path=get_lg_path(),
        num_iterations=1,
        num_processes=32,
        quantization_grid_center=0.03,
        quantization_grid_step=1.1,
        quantization_grid_num_steps=64,
        learning_rate=1e-1,
        num_epochs=2000,
        do_adam=True,
        use_cpp_counting_implementation=True,
        num_processes_optimization=2,
    )["learned_rate_matrix_path"]
    learned_rate_matrix = read_rate_matrix(learned_rate_matrix_path).to_numpy()

    lg = read_rate_matrix(get_lg_path()).to_numpy()

    # Now compare matrices
    print("LG matrix:")
    print(lg[:3, :3])
    print("Learned rate matrix:")
    print(learned_rate_matrix[:3, :3])
