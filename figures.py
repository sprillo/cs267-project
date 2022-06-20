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
import tempfile
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from src.markov_chain import compute_stationary_distribution, matrix_exponential, normalized, compute_mutation_rate, chain_product
from src.io import write_count_matrices, read_log_likelihood, write_probability_distribution, read_probability_distribution, write_rate_matrix, read_msa, write_msa, read_site_rates, write_site_rates
from src.estimation import quantized_transitions_mle
from matplotlib.colors import LogNorm
from src.estimation import jtt_ipw
from src.phylogeny_estimation import fast_tree
from src.types import PhylogenyEstimatorType
from src.evaluation import compute_log_likelihoods, create_maximal_matching_contact_map

from src import caching, cherry_estimator, cherry_estimator_coevolution
from src.benchmarking.pfam_15k import (
    compute_contact_maps,
    get_families,
    get_families_within_cutoff,
    simulate_ground_truth_data_coevolution,
    simulate_ground_truth_data_single_site,
    subsample_pfam_15k_msas,
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
    get_aa_coevolution_mask_path,
    get_equ_path,
    get_equ_x_equ_path,
    get_jtt_path,
    get_wag_path,
    get_lg_path,
    get_lg_x_lg_path,
    normalized,
)
from src.phylogeny_estimation import gt_tree_estimator
import src.utils as utils

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


##### Draft figures (still need work) #####

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
            ("Cherry__1__1e-1__2000", "Cherry\n1st iteration"),
            ("Cherry__2__1e-1__2000", "Cherry\n2nd iteration"),
            ("Cherry__3__1e-1__2000", "Cherry\n3rd iteration"),
            ("Cherry__4__1e-1__2000", "Cherry\n4th iteration"),
            ("Cherry__5__1e-1__2000", "Cherry\n5th iteration"),
            ("Cherry__6__1e-1__2000", "Cherry\n6th iteration"),
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

##### Single site experiments #####

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
                os.path.join(output_rate_matrix_dir, "Q_best.txt")
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
    quantized.
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

            learned_rate_matrix_path = os.path.join(
                cherry_estimator_res["rate_matrix_dir_0"],
                "Q_best.txt"
            )
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


def fig_convergence_on_large_data_single_site__variance():
    """
    We study the variance of the max absolute error as the random seed changes.

    This aims to shed more light into fig_convergence_on_large_data_single_site.
    """
    caching.set_cache_dir("_cache_benchmarking")
    caching.set_hash_len(64)

    output_image_dir = "images/fig_convergence_on_large_data_single_site__variance"
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
    random_seed_grid = list(range(10))

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
        for random_seed in random_seed_grid:
            num_families_train = 15051
            num_families_test = 0
            
            quantization_grid_center = 0.03
            quantization_grid_step = 1.1
            quantization_grid_num_steps = 64
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
                num_epochs=32768,
                do_adam=True,
                use_cpp_counting_implementation=use_cpp_implementation,
                num_processes_optimization=2,
            )

            learned_rate_matrix_path = os.path.join(
                cherry_estimator_res["rate_matrix_dir_0"],
                "Q_best.txt"
            )
            learned_rate_matrix = read_rate_matrix(learned_rate_matrix_path)
            learned_rate_matrix = learned_rate_matrix.to_numpy()
            res = relative_errors(Q_numpy, learned_rate_matrix)
            result_tuples.append((learning_rate, random_seed, np.mean(res), np.median(res), np.max(res)))
            res_2d_row['mean'].append(np.mean(res))
            res_2d_row['median'].append(np.median(res))
            res_2d_row['max'].append(np.max(res))
        res_2d['mean'].append(res_2d_row['mean'])
        res_2d['median'].append(res_2d_row['median'])
        res_2d['max'].append(res_2d_row['max'])

    res = pd.DataFrame(
        result_tuples,
        columns=["learning_rate", "random_seed", "mean_relative_error", "median_relative_error", "max_relative_error"]
    )
    # print(res)

    for metric_name in ['max', 'median']:
        sns.heatmap(
            np.array(res_2d[metric_name]).T,
            yticklabels=random_seed_grid,
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
        plt.ylabel("random seed")
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

        learned_rate_matrix_path = os.path.join(
            cherry_estimator_res["rate_matrix_dir_0"],
            "Q_best.txt"
        )

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

            learned_rate_matrix_path = os.path.join(
                cherry_estimator_res["rate_matrix_dir_0"],
                "Q_best.txt"
            )
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


def fig_jtt_ipw_single_site():
    """
    We show that initializing with JTT-IPW speeds up convergence
    over EQU and random initialization.
    """
    caching.set_cache_dir("_cache_benchmarking")
    caching.set_hash_len(64)

    output_image_dir = "images/fig_jtt_ipw_single_site"
    if not os.path.exists(output_image_dir):
        os.makedirs(output_image_dir)

    initialization_grid = [
        "jtt-ipw",
        "equ",
        "random",
    ]
    learning_rate = 1e-1
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

    for initialization in initialization_grid:
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
                optimizer_initialization=initialization,
            )

            learned_rate_matrix_path = os.path.join(
                cherry_estimator_res["rate_matrix_dir_0"],
                "Q_best.txt"
            )
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
        columns=["initialization", "num_epochs", "mean_relative_error", "median_relative_error", "max_relative_error"]
    )
    # print(res)

    for metric_name in ['max', 'median']:
        sns.heatmap(
            np.array(res_2d[metric_name]).T,
            yticklabels=num_epochs_grid,
            xticklabels=initialization_grid,
            cmap="YlGnBu",  # "RdBu_r"
            annot=True,
            annot_kws={"size": 6},
            fmt=".1",
            # vmin=0,
            # vmax=vmax,
            # center=center,
            norm=LogNorm(),
        )
        plt.xlabel("initialization")
        plt.ylabel("number of epochs")
        plt.title(f"{metric_name} relative error")
        # plt.gcf().set_size_inches(16, 16)
        plt.tight_layout()
        plt.savefig(f"{output_image_dir}/heatmap_{metric_name}.png", dpi=300)
        plt.close()


##### Pair of site experiments #####

def fig_convergence_on_infinite_data_pair_site():
    """
    We show that on "infinite" pair-of-site data, the pytorch optimizer
    converges to the solution for a variety of learning rates, and we identify
    the optimal learning rate to be 0.1: small - so as to be numerically
    stable - but not too small - so as to converge fast.
    """
    caching.set_cache_dir("_cache_benchmarking")
    caching.set_hash_len(64)

    output_image_dir = "images/fig_convergence_on_infinite_data_pair_site"
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
    ]

    rate_matrix_path = get_lg_x_lg_path()
    Q_df = read_rate_matrix(rate_matrix_path)
    Q_numpy = Q_df.to_numpy()
    mask_path = get_aa_coevolution_mask_path()
    mask_matrix = read_mask_matrix(mask_path)

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
            initialization_path = get_equ_x_equ_path()
            # Run the Adam optimizer.
            output_rate_matrix_dir = quantized_transitions_mle(
                count_matrices_path=count_matrices_path,
                initialization_path=initialization_path,
                mask_path=mask_path,
                stationary_distribution_path=None,
                rate_matrix_parameterization="pande_reversible",
                device="cpu",
                learning_rate=learning_rate,
                num_epochs=num_epochs,
                do_adam=True,
                OMP_NUM_THREADS=8,
                OPENBLAS_NUM_THREADS=8,
            )["output_rate_matrix_dir"]

            learned_rate_matrix = read_rate_matrix(
                os.path.join(output_rate_matrix_dir, "Q_best.txt")
            ).to_numpy()

            res = relative_errors(Q_numpy, learned_rate_matrix, mask_matrix)
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


def fig_convergence_on_large_data_pair_site():
    """
    We show that on pair-site data simulated on top of real trees, the pytorch
    optimizer converges to the solution for a variety of learning rates, and we
    validate the optimal learning rate to be 0.1. This figure provides more
    information than:
    fig_convergence_on_infinite_data_pair_site
    in that the branch lengths come from real data, and they are also getting
    quantized.
    """
    caching.set_cache_dir("_cache_benchmarking")
    caching.set_hash_len(64)

    output_image_dir = "images/fig_convergence_on_large_data_pair_site"
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
    ]

    minimum_distance_for_nontrivial_contact = (
        7
    )
    angstrom_cutoff = 8.0

    Q_numpy = read_rate_matrix(get_lg_x_lg_path()).to_numpy()
    mask_matrix = read_rate_matrix(get_aa_coevolution_mask_path()).to_numpy()

    num_processes = 8
    num_sequences = (
        1024
    )
    num_rate_categories = (
        20
    )

    result_tuples = []
    res_2d = {'mean': [], 'median': [], 'max': []}

    for learning_rate in learning_rate_grid:
        msg = f"***** learning_rate = {learning_rate} *****"
        print("*" * len(msg))
        print(msg)
        print("*" * len(msg))
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
                coevolution_mask_path=get_aa_coevolution_mask_path(),
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
                do_adam=True,
                use_cpp_counting_implementation=use_cpp_implementation,
                num_processes_optimization=8,
            )

            learned_rate_matrix_path = os.path.join(
                cherry_estimator_res["rate_matrix_dir_0"],
                "Q_best.txt"
            )
            if num_epochs == 8192:
                print(f"learned_rate_matrix_path = {learned_rate_matrix_path}")
            learned_rate_matrix = read_rate_matrix(learned_rate_matrix_path)
            learned_rate_matrix = learned_rate_matrix.to_numpy()
            res = relative_errors(Q_numpy, learned_rate_matrix, mask_matrix=mask_matrix)
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


def fig_pair_site_quantization_error():
    """
    We show that ~100 quantization points (geometric increments of 10%) is
    enough.
    """
    output_image_dir = "images/fig_pair_site_quantization_error"
    if not os.path.exists(output_image_dir):
        os.makedirs(output_image_dir)

    num_processes = 8
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
    do_adam = True
    use_cpp_implementation = (
        True
    )
    minimum_distance_for_nontrivial_contact = (
        7
    )
    num_epochs = 500
    angstrom_cutoff = 8.0

    caching.set_cache_dir("_cache_benchmarking")
    caching.set_hash_len(64)

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
            num_processes_optimization=8,
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
        ticks = [0.0003, 0.003, 0.03, 0.3, 3.0]
        plt.xticks(np.log(ticks), ticks)
        plt.savefig(f"{output_image_dir}/count_matrices_{i}", dpi=300)
        plt.close()

        learned_rate_matrix_path = os.path.join(
            cherry_estimator_res["rate_matrix_dir_0"],
            "Q_best.txt"
        )
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


##### Real-data experiments #####

@caching.cached_computation(
    output_dirs=["output_probability_distribution_dir"],
)
def get_stationary_distribution(
    rate_matrix_path: str,
    output_probability_distribution_dir: Optional[str] = None,
):
    rate_matrix = read_rate_matrix(rate_matrix_path)
    pi = compute_stationary_distribution(rate_matrix.to_numpy())
    write_probability_distribution(pi, rate_matrix.index, os.path.join(output_probability_distribution_dir, "result.txt"))


@caching.cached_computation(
    output_dirs=["output_rate_matrix_dir"],
)
def normalize_rate_matrix(
    rate_matrix_path: str,
    new_rate: float,
    output_rate_matrix_dir: Optional[str] = None,
):
    rate_matrix = read_rate_matrix(rate_matrix_path)
    normalized_rate_matrix = new_rate * normalized(rate_matrix.to_numpy())
    write_rate_matrix(normalized_rate_matrix, rate_matrix.index, os.path.join(output_rate_matrix_dir, "result.txt"))


@caching.cached_computation(
    output_dirs=["output_rate_matrix_dir"],
)
def chain_product_cached(
    rate_matrix_1_path: str,
    rate_matrix_2_path: str,
    output_rate_matrix_dir: Optional[str] = None,
):
    rate_matrix_1 = read_rate_matrix(rate_matrix_1_path)
    rate_matrix_2 = read_rate_matrix(rate_matrix_2_path)
    res = chain_product(rate_matrix_1.to_numpy(), rate_matrix_2.to_numpy())
    if list(rate_matrix_1.index) != list(rate_matrix_2.index):
        raise Exception(f"Double-check that the states are being computed correctly in the code.")
    states = [state_1 + state_2 for state_1 in rate_matrix_1.index for state_2 in rate_matrix_2.index]
    write_rate_matrix(res, states, os.path.join(output_rate_matrix_dir, "result.txt"))


def evaluate_single_site_model_on_held_out_msas(
    msa_dir: str,
    families: List[str],
    rate_matrix_path: str,
    num_processes: int,
    tree_estimator: PhylogenyEstimatorType,
):
    """
    Evaluate a reversible single-site model on held out msas.
    """
    # First estimate the trees
    tree_estimator_output_dirs = tree_estimator(
        msa_dir=msa_dir,
        families=families,
        rate_matrix_path=rate_matrix_path,
        num_processes=num_processes,
    )
    tree_dir = tree_estimator_output_dirs["output_tree_dir"]
    site_rates_dir = tree_estimator_output_dirs["output_site_rates_dir"]
    output_probability_distribution_dir = get_stationary_distribution(
        rate_matrix_path=rate_matrix_path,
    )["output_probability_distribution_dir"]
    pi_1_path = os.path.join(output_probability_distribution_dir, "result.txt")
    output_likelihood_dir = compute_log_likelihoods(
        tree_dir=tree_dir,
        msa_dir=msa_dir,
        site_rates_dir=site_rates_dir,
        contact_map_dir=None,
        families=families,
        amino_acids=utils.get_amino_acids(),
        pi_1_path=pi_1_path,
        Q_1_path=rate_matrix_path,
        reversible_1=True,
        device_1="cpu",
        pi_2_path=None,
        Q_2_path=None,
        reversible_2=None,
        device_2=None,
        num_processes=num_processes,
        use_cpp_implementation=False,
        OMP_NUM_THREADS=1,
        OPENBLAS_NUM_THREADS=1,
    )["output_likelihood_dir"]
    lls = []
    for family in families:
        ll = read_log_likelihood(os.path.join(output_likelihood_dir, f"{family}.txt"))
        lls.append(ll[0])
    return np.sum(lls)


def evaluate_pair_site_model_on_held_out_msas(
    msa_dir: str,
    contact_map_dir: str,
    families: List[str],
    rate_matrix_1_path: str,
    rate_matrix_2_path: str,
    num_processes: int,
    tree_estimator: PhylogenyEstimatorType,
):
    """
    Evaluate a reversible single-site model and coevolution model on held out
    msas.
    """
    # First estimate the trees
    tree_estimator_output_dirs = tree_estimator(
        msa_dir=msa_dir,
        families=families,
        rate_matrix_path=rate_matrix_1_path,
        num_processes=num_processes,
    )
    tree_dir = tree_estimator_output_dirs["output_tree_dir"]
    site_rates_dir = tree_estimator_output_dirs["output_site_rates_dir"]
    output_probability_distribution_1_dir = get_stationary_distribution(
        rate_matrix_path=rate_matrix_1_path,
    )["output_probability_distribution_dir"]
    pi_1_path = os.path.join(output_probability_distribution_1_dir, "result.txt")
    output_probability_distribution_2_dir = get_stationary_distribution(
        rate_matrix_path=rate_matrix_2_path,
    )["output_probability_distribution_dir"]
    pi_2_path = os.path.join(output_probability_distribution_2_dir, "result.txt")
    output_likelihood_dir = compute_log_likelihoods(
        tree_dir=tree_dir,
        msa_dir=msa_dir,
        site_rates_dir=site_rates_dir,
        contact_map_dir=contact_map_dir,
        families=families,
        amino_acids=utils.get_amino_acids(),
        pi_1_path=pi_1_path,
        Q_1_path=rate_matrix_1_path,
        reversible_1=True,
        device_1="cpu",
        pi_2_path=pi_2_path,
        Q_2_path=rate_matrix_2_path,
        reversible_2=True,
        device_2="cpu",
        num_processes=num_processes,
        use_cpp_implementation=False,
        OMP_NUM_THREADS=1,
        OPENBLAS_NUM_THREADS=1,
    )["output_likelihood_dir"]
    lls = []
    for family in families:
        ll = read_log_likelihood(os.path.join(output_likelihood_dir, f"{family}.txt"))
        lls.append(ll[0])
    return np.sum(lls)


def fig_pfam15k():
    """
    We use 12K families for training and 3K for testing.

    We fit trees with LG and then learn a single-site model (Cherry) and a
    coevolution model (Cherry2).

    We next fit trees on the 3K test families and compute the likelihood under:
    - JTT
    - WAG
    - LG
    - Cherry
    - Cherry + Cherry2
    """
    caching.set_cache_dir("_cache_benchmarking")
    caching.set_hash_len(64)

    PFAM_15K_MSA_DIR = "input_data/a3m"
    PFAM_15K_PDB_DIR = "input_data/pdb"

    num_processes = 32
    num_sequences = 1024
    num_families_train = 12000
    num_families_test = 3000
    train_test_split_seed = 0
    use_cpp_implementation = True
    num_rate_categories = 1  #  TODO: 4  To be fair with LG, since LG was trained with 4 rate categories
    use_Q_best = True
    angstrom_cutoff = 8.0
    minimum_distance_for_nontrivial_contact = 7
    use_maximal_matching = True

    families_all = get_families(
        pfam_15k_msa_dir=PFAM_15K_MSA_DIR,
    )
    np.random.seed(train_test_split_seed)
    np.random.shuffle(families_all)

    families_train = sorted(families_all[:num_families_train])
    if num_families_test == 0:
        families_test = []
    else:
        families_test = sorted(families_all[-num_families_test:])
    if num_families_train + num_families_test > len(families_all):
        raise Exception(f"Training and testing set would overlap!")
    assert(len(set(families_train + families_test)) == len(families_train) + len(families_test))

    # Subsample the MSAs
    msa_dir_train = subsample_pfam_15k_msas(
        pfam_15k_msa_dir=PFAM_15K_MSA_DIR,
        num_sequences=num_sequences,
        families=families_train,
        num_processes=num_processes,
    )["output_msa_dir"]

    # Run the cherry method using FastTree tree estimator
    cherry_path = cherry_estimator(
        msa_dir=msa_dir_train,
        families=families_train,
        tree_estimator=partial(
            fast_tree,
            num_rate_categories=num_rate_categories,
        ),
        initial_tree_estimator_rate_matrix_path=get_lg_path(),
        num_iterations=1,
        num_processes=num_processes,
        quantization_grid_center=0.03,
        quantization_grid_step=1.1,
        quantization_grid_num_steps=64,
        learning_rate=1e-1,
        num_epochs=2000,
        do_adam=True,
        use_cpp_counting_implementation=use_cpp_implementation,
        num_processes_optimization=2,
        optimizer_return_best_iter=use_Q_best,
    )["learned_rate_matrix_path"]
    cherry = read_rate_matrix(
        cherry_path
    ).to_numpy()

    lg = read_rate_matrix(get_lg_path()).to_numpy()

    # Now compare matrices
    print("Cherry topleft 3x3 corner:")
    print(cherry[:3, :3])
    print("LG topleft 3x3 corner:")
    print(lg[:3, :3])

    # Subsample the MSAs
    msa_dir_test = subsample_pfam_15k_msas(
        pfam_15k_msa_dir=PFAM_15K_MSA_DIR,
        num_sequences=num_sequences,
        families=families_test,
        num_processes=num_processes,
    )["output_msa_dir"]

    for rate_matrix_name, rate_matrix_path in [
        ("EQU", get_equ_path()),
        ("JTT", get_jtt_path()),
        ("WAG", get_wag_path()),
        ("LG", get_lg_path()),
        ("Cherry", cherry_path),
    ]:
        print(f"***** Evaluating: {rate_matrix_name} ({num_rate_categories} rate categories) *****")
        ll = evaluate_single_site_model_on_held_out_msas(
            msa_dir=msa_dir_test,
            families=families_test,
            rate_matrix_path=rate_matrix_path,
            num_processes=num_processes,
            tree_estimator=partial(
                fast_tree,
                num_rate_categories=num_rate_categories,
            ),
        )
        print(f"ll for {rate_matrix_name} = {ll}")

    # Run the single-site cherry method *ONLY ON CONTACTING SITES*
    contact_map_dir_train = compute_contact_maps(
        pfam_15k_pdb_dir=PFAM_15K_PDB_DIR,
        families=families_train,
        angstrom_cutoff=angstrom_cutoff,
        num_processes=num_processes,
    )["output_contact_map_dir"]

    mdnc = minimum_distance_for_nontrivial_contact
    cherry_contact_path = cherry_estimator(
        msa_dir=msa_dir_train,
        families=families_train,
        tree_estimator=partial(
            fast_tree,
            num_rate_categories=num_rate_categories,
        ),
        initial_tree_estimator_rate_matrix_path=get_lg_path(),
        num_iterations=1,
        num_processes=num_processes,
        quantization_grid_center=0.03,
        quantization_grid_step=1.1,
        quantization_grid_num_steps=64,
        learning_rate=1e-1,
        num_epochs=2000,
        do_adam=True,
        use_cpp_counting_implementation=use_cpp_implementation,
        num_processes_optimization=2,
        optimizer_return_best_iter=use_Q_best,
        use_only_contacting_sites_in_optimizer=True,
        contact_map_dir=contact_map_dir_train,
        minimum_distance_for_nontrivial_contact=mdnc,
    )["learned_rate_matrix_path"]
    cherry_contact = read_rate_matrix(
        cherry_contact_path
    ).to_numpy()
    print("Cherry contact topleft 3x3 corner:")
    print(cherry_contact[:3, :3])

    cherry_contact_squared_path = os.path.join(
        chain_product_cached(
            rate_matrix_1_path=cherry_contact_path,
            rate_matrix_2_path=cherry_contact_path,
        )["output_rate_matrix_dir"],
        "result.txt"
    )

    # cherry_squared_path = os.path.join(
    #     chain_product_cached(
    #         rate_matrix_1_path=cherry_path,
    #         rate_matrix_2_path=cherry_path,
    #     )["output_rate_matrix_dir"],
    #     "result.txt"
    # )

    ##### Now estimate and evaluate the coevolution model #####
    cherry_2_path = cherry_estimator_coevolution(
        msa_dir=msa_dir_train,
        contact_map_dir=contact_map_dir_train,
        minimum_distance_for_nontrivial_contact=mdnc,
        coevolution_mask_path=get_aa_coevolution_mask_path(),
        families=families_train,
        tree_estimator=partial(
            fast_tree,
            num_rate_categories=num_rate_categories,
        ),
        initial_tree_estimator_rate_matrix_path=get_lg_path(),
        num_processes=num_processes,
        quantization_grid_center=0.03,
        quantization_grid_step=1.1,
        quantization_grid_num_steps=64,
        learning_rate=1e-1,
        num_epochs=500,
        do_adam=True,
        use_cpp_counting_implementation=use_cpp_implementation,
        num_processes_optimization=8,
        optimizer_return_best_iter=use_Q_best,
        use_maximal_matching=use_maximal_matching,
    )["learned_rate_matrix_path"]

    # cherry_2_normalized_to_rate_of_2_x_cherry__path = os.path.join(
    #     normalize_rate_matrix(
    #         rate_matrix_path=cherry_2_path,
    #         new_rate=2.0 * compute_mutation_rate(read_rate_matrix(cherry_path).to_numpy())
    #     )["output_rate_matrix_dir"],
    #     "result.txt",
    # )

    # cherry_2_normalized_to_rate_2__path = os.path.join(
    #     normalize_rate_matrix(
    #         rate_matrix_path=cherry_2_path,
    #         new_rate=2.0
    #     )["output_rate_matrix_dir"],
    #     "result.txt",
    # )

    contact_map_dir_test = compute_contact_maps(
        pfam_15k_pdb_dir=PFAM_15K_PDB_DIR,
        families=families_test,
        angstrom_cutoff=angstrom_cutoff,
        num_processes=num_processes,
    )["output_contact_map_dir"]
    contact_map_dir_test = create_maximal_matching_contact_map(
        i_contact_map_dir=contact_map_dir_test,
        families=families_test,
        minimum_distance_for_nontrivial_contact=mdnc,
        num_processes=num_processes,
    )["o_contact_map_dir"]

    for rate_matrix_2_name, rate_matrix_2_path in [
        # ("Cherry squared", cherry_squared_path),  # DEBUG: Should give same result as single-site Cherry
        ("Cherry contact squared", cherry_contact_squared_path),  # Fair baseline to compare coevolution likelihood against.
        ("Cherry2", cherry_2_path),
        # ("Cherry2 normalized to rate of 2 x Cherry", cherry_2_normalized_to_rate_of_2_x_cherry__path),  # Pointless: Cherry2 is the right thing to do
        # ("Cherry2 normalized to rate of 2.0", cherry_2_normalized_to_rate_2__path),  # Pointless: Cherry2 is the right thing to do
    ]:
        print(f"***** Evaluating: {rate_matrix_2_name} ({num_rate_categories} rate categories) *****")
        if num_rate_categories > 1:
            print("***** TODO: It is unclear to me whether the evaluation makes sense when num_rate_categories > 1, because of unidentifiability between site rates and branch lengths, and the fact that we are using branch lengths as the time unit for the coevolution models. In other words, we might be unfair with the coevolution model *****")
        ll = evaluate_pair_site_model_on_held_out_msas(
            msa_dir=msa_dir_test,
            contact_map_dir=contact_map_dir_test,
            families=families_test,
            rate_matrix_1_path=cherry_path,
            rate_matrix_2_path=rate_matrix_2_path,
            num_processes=num_processes,
            tree_estimator=partial(
                fast_tree,
                num_rate_categories=num_rate_categories,
            ),
        )
        print(f"ll for {rate_matrix_2_name} = {ll}")
