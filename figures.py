from src.benchmarking.pfam_15k import (
    simulate_ground_truth_data_single_site,
    get_families_within_cutoff,
    simulate_ground_truth_data_coevolution,
)
from src.markov_chain import (
    get_equ_path,
    normalized,
    get_lg_path,
    get_lg_x_lg_path,
)
from src import cherry_estimator, cherry_estimator_coevolution
from src import caching
from src.evaluation import (
    l_infty_norm,
    rmse,
    mre,
    mean_relative_error,
    relative_errors,
    plot_rate_matrix_predictions,
)
from src.io import read_rate_matrix, read_count_matrices, read_mask_matrix
from src.phylogeny_estimation import gt_tree_estimator
from functools import partial
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from typing import Dict, List

PFAM_15K_MSA_DIR = "input_data/a3m"
PFAM_15K_PDB_DIR = "input_data/pdb"


def fig_single_site_quantization_error():
    """
    TODO: Use all 15051 families for this, to reduce finite sample size bias.
    """
    output_image_dir = "images/fig_single_site_quantization_error"
    if not os.path.exists(output_image_dir):
        os.makedirs(output_image_dir)

    num_processes = 32  # Leave some processes open for other work.
    num_sequences = (
        1024  # We use a lot of sequences per family to take variance to 0.
    )
    num_rate_categories = (
        20  # Only used for creating the GT trees with FastTree.
    )

    # TODO: For co-evolution, we may need more num_families_train?
    ##### To use just the 1448 families with ~210 sites:
    min_num_sites = 190  # We make sure that our simulated MSAs all have roughly the same number of sites, and equal to the median number of sites over the whole dataset (which is 210)
    max_num_sites = 230  # We make sure that our simulated MSAs all have roughly the same number of sites, and equal to the median number of sites over the whole dataset (which is 210)
    min_num_sequences = num_sequences  # We only select families with at least 1024 sequences, such that all the simulated MSAs have exactly the same size. (Even though we don't explore this dimension here, in other experiments we do, and it is nice to be able to compare the results in this figure to those of the other figures, so we use the same training sets across figures if possible.)
    max_num_sequences = 1000000  # We don't want to filter families with more that 1024 sequences, since they will be subsapled later down to 1024.
    num_families_train = 1024  # We fix the number of families to a large number of 1024, to take variance to 0.
    num_families_test = 0  # We just evaluate l_infty_norm and rmse, we don't look at held out likelihood (we are less interested in held out since we know the ground truth rate matrix)
    ##### To use all 15051 families:
    # min_num_sites = 0  # We make sure that our simulated MSAs all have roughly the same number of sites, and equal to the median number of sites over the whole dataset (which is 210)
    # max_num_sites = 1000000  # We make sure that our simulated MSAs all have roughly the same number of sites, and equal to the median number of sites over the whole dataset (which is 210)
    # min_num_sequences = 0  # We only select families with at least 1024 sequences, such that all the simulated MSAs have exactly the same size. (Even though we don't explore this dimension here, in other experiments we do, and it is nice to be able to compare the results in this figure to those of the other figures, so we use the same training sets across figures if possible.)
    # max_num_sequences = 1000000  # We don't want to filter families with more that 1024 sequences, since they will be subsapled later down to 1024.
    # num_families_train = 15051  # We fix the number of families to a large number of 1024, to take variance to 0.
    # num_families_test = 0  # We just evaluate l_infty_norm and rmse, we don't look at held out likelihood (we are less interested in held out since we know the ground truth rate matrix)

    quantization_grid_center = None  # This we will iterate over
    quantization_grid_step = None  # This we will iterate over
    quantization_grid_num_steps = None  # This we will iterate over
    random_seed = 0  # We fix this, since the number of sequences and sites is large enough to ensure convergence of variance to 0. I.e. there is no variance in this experiment, so we can just set the random seed to 0.
    learning_rate = 3e-2  # We use a highly precise optimizer for this
    num_epochs = 2000  # We use a highly precise optimizer for this
    do_adam = True  # We use a highly precise optimizer for this
    use_cpp_implementation = (
        True  # For simulating MSAs and counting, super fast.
    )

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
            pfam_15k_msa_dir=PFAM_15K_MSA_DIR,  # We use the PFAM 15K dataset as reference for the simulated MSA sizes
            min_num_sites=min_num_sites,  # We make sure that our simulated MSAs all have roughly the same number of sites, and equal to the median number of sites over the whole dataset (which is 210)
            max_num_sites=max_num_sites,  # We make sure that our simulated MSAs all have roughly the same number of sites, and equal to the median number of sites over the whole dataset (which is 210)
            min_num_sequences=min_num_sequences,  # We only select families with at least 1024 sequences, such that all the simulated MSAs have exactly the same size. (Even though we don't explore this dimension here, in other experiments we do, and it is nice to be able to compare the results in this figure to those of the other figures, so we use the same training sets across figures if possible.)
            max_num_sequences=max_num_sequences,  # We don't want to filter families with more that 1024 sequences, since they will be subsapled later down to 1024.
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
            pfam_15k_msa_dir=PFAM_15K_MSA_DIR,  # We use the PFAM 15K as the reference to build the GT trees.
            num_sequences=num_sequences,  # 1024
            families=families_train + families_test,  # 1024 families
            num_rate_categories=num_rate_categories,  # 20, since this is the standard number of rate categories to use.
            num_processes=num_processes,
            random_seed=random_seed,  # This will be fixed to 0, since the dataset is large, making variance vanish and bias show itself.
            use_cpp_simulation_implementation=use_cpp_implementation,  # Fastest!
        )

        # Now run the cherry method with access to GT trees
        cherry_estimator_res = cherry_estimator(
            msa_dir=msa_dir,  # Simulated MSAs
            families=families_train,  # 1024
            tree_estimator=partial(  # We use the GT tree estimator
                gt_tree_estimator,
                gt_tree_dir=gt_tree_dir,
                gt_site_rates_dir=gt_site_rates_dir,
                gt_likelihood_dir=gt_likelihood_dir,
                num_rate_categories=num_rate_categories,  # This doesn't matter because we are using GT Trees
            ),
            initial_tree_estimator_rate_matrix_path=get_equ_path(),  # This doesn't matter because we are using GT Trees
            num_iterations=1,  # We use GT trees, so no need to iterate.
            num_processes=num_processes,
            quantization_grid_center=quantization_grid_center,  # This is what we iterate over
            quantization_grid_step=quantization_grid_step,  # This is what we iterate over
            quantization_grid_num_steps=quantization_grid_num_steps,  # This is what we iterate over
            learning_rate=learning_rate,
            num_epochs=num_epochs,
            do_adam=do_adam,
            use_cpp_counting_implementation=use_cpp_implementation,  # Fastest!
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

        learned_rate_matrix = read_rate_matrix(learned_rate_matrix_path)

        # TODO: Don't normalize!
        learned_rate_matrix = learned_rate_matrix.to_numpy()
        # learned_rate_matrix = normalized(learned_rate_matrix.to_numpy())
        Qs.append(learned_rate_matrix)

        lg = read_rate_matrix(get_lg_path()).to_numpy()  # GT rate matrix

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
    yticks = [np.log(10**i) for i in range(-5, 2)]
    ytickslabels = [f"$10^{{{i + 2}}}$" for i in range(-5, 2)]
    plt.grid()
    plt.ylabel("relative error")
    plt.yticks(yticks, ytickslabels)
    for i, ys in enumerate(yss_relative_errors):
        ys = np.array(ys)
        label = "{:.1f}%".format(100 * np.median(ys))
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
            color="black",
            fontsize=12,
        )  # horizontal alignment can be left, right or center
    plt.title(
        "Distribution of relative error as quantization improves\n(median also reported)"
    )
    plt.tight_layout()
    plt.savefig(f"{output_image_dir}/violin_plot", dpi=300)
    plt.close()


def fig_pair_site_quantization_error():
    # TODO: Use all 15051 families
    # TODO: Do NOT normalize the learned rate matrix!
    output_image_dir = "images/fig_pair_site_quantization_error"
    if not os.path.exists(output_image_dir):
        os.makedirs(output_image_dir)

    num_processes = 32  # Leave some processes open for other work.
    num_sequences = (
        1024  # We use a lot of sequences per family to take variance to 0.
    )
    num_rate_categories = (
        20  # Only used for craeting the GT trees with FastTree.
    )

    # TODO: For co-evolution, we may need more num_families_train?
    ##### To use just the 1448 families with ~210 sites:
    min_num_sites = 190  # We make sure that our simulated MSAs all have roughly the same number of sites, and equal to the median number of sites over the whole dataset (which is 210)
    max_num_sites = 230  # We make sure that our simulated MSAs all have roughly the same number of sites, and equal to the median number of sites over the whole dataset (which is 210)
    min_num_sequences = num_sequences  # We only select families with at least 1024 sequences, such that all the simulated MSAs have exactly the same size. (Even though we don't explore this dimension here, in other experiments we do, and it is nice to be able to compare the results in this figure to those of the other figures, so we use the same training sets across figures if possible.)
    max_num_sequences = 1000000  # We don't want to filter families with more that 1024 sequences, since they will be subsapled later down to 1024.
    num_families_train = 1024  # We fix the number of families to a large number of 1024, to take variance to 0.
    num_families_test = 0  # We just evaluate l_infty_norm and rmse, we don't look at held out likelihood (we are less interested in held out since we know the ground truth rate matrix)
    ##### To use all 15051 families:
    # min_num_sites = 0  # We make sure that our simulated MSAs all have roughly the same number of sites, and equal to the median number of sites over the whole dataset (which is 210)
    # max_num_sites = 1000000  # We make sure that our simulated MSAs all have roughly the same number of sites, and equal to the median number of sites over the whole dataset (which is 210)
    # min_num_sequences = 0  # We only select families with at least 1024 sequences, such that all the simulated MSAs have exactly the same size. (Even though we don't explore this dimension here, in other experiments we do, and it is nice to be able to compare the results in this figure to those of the other figures, so we use the same training sets across figures if possible.)
    # max_num_sequences = 1000000  # We don't want to filter families with more that 1024 sequences, since they will be subsapled later down to 1024.
    # num_families_train = 15051  # We fix the number of families to a large number of 1024, to take variance to 0.
    # num_families_test = 0  # We just evaluate l_infty_norm and rmse, we don't look at held out likelihood (we are less interested in held out since we know the ground truth rate matrix)

    quantization_grid_center = None  # This we will iterate over
    quantization_grid_step = None  # This we will iterate over
    quantization_grid_num_steps = None  # This we will iterate over
    random_seed = 0  # We fix this, since the number of sequences and sites is large enough to ensure convergence of variance to 0. I.e. there is no variance in this experiment, so we can just set the random seed to 0.
    learning_rate = 3e-2  # 3 * (1e-2)
    do_adam = True
    use_cpp_implementation = (
        True  # For simulating MSAs and counting, super fast.
    )
    minimum_distance_for_nontrivial_contact = (
        7  # We use the standard of 7 position away.
    )
    num_epochs = 200  # An accurate optimizer (for single-site model). Only 200 epochs since slower.
    angstrom_cutoff = 8.0  # Standard cutoff for determining contacts.

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
        # (0.06, 1.024, 256),
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
            pfam_15k_msa_dir=PFAM_15K_MSA_DIR,  # We use the PFAM 15K dataset as reference for the simulated MSA sizes
            min_num_sites=min_num_sites,  # We make sure that our simulated MSAs all have roughly the same number of sites, and equal to the median number of sites over the whole dataset (which is 210)
            max_num_sites=max_num_sites,  # We make sure that our simulated MSAs all have roughly the same number of sites, and equal to the median number of sites over the whole dataset (which is 210)
            min_num_sequences=min_num_sequences,  # We only select families with at least 1024 sequences, such that all the simulated MSAs have exactly the same size. (Even though we don't explore this dimension here, in other experiments we do, and it is nice to be able to compare the results in this figure to those of the other figures, so we use the same training sets across figures if possible.)
            max_num_sequences=max_num_sequences,  # We don't want to filter families with more that 1024 sequences, since they will be subsapled later down to 1024.
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
            pfam_15k_msa_dir=PFAM_15K_MSA_DIR,  # We use the PFAM 15K as the reference to build the GT trees.
            pfam_15k_pdb_dir=PFAM_15K_PDB_DIR,
            minimum_distance_for_nontrivial_contact=minimum_distance_for_nontrivial_contact,
            angstrom_cutoff=angstrom_cutoff,
            num_sequences=num_sequences,  # 1024
            families=families_all,  # 1024 families
            num_rate_categories=num_rate_categories,  # 20, since this is the standard number of rate categories to use.
            num_processes=num_processes,
            random_seed=random_seed,  # This will be fixed to 0, since the dataset is large, making variance vanish and bias show itself.
            use_cpp_simulation_implementation=use_cpp_implementation,  # Fastest!
        )

        # Now run the cherry method with access to GT trees
        cherry_estimator_res = cherry_estimator_coevolution(
            msa_dir=msa_dir,  # Simulated MSAs
            contact_map_dir=contact_map_dir,  # Synthetic contact maps.
            minimum_distance_for_nontrivial_contact=minimum_distance_for_nontrivial_contact,
            coevolution_mask_path="data/mask_matrices/aa_coevolution_mask.txt",
            families=families_train,  # 1024
            tree_estimator=partial(  # We use the GT tree estimator
                gt_tree_estimator,
                gt_tree_dir=gt_tree_dir,
                gt_site_rates_dir=gt_site_rates_dir,
                gt_likelihood_dir=gt_likelihood_dir,
                num_rate_categories=num_rate_categories,  # This doesn't matter because we are using GT Trees
            ),
            initial_tree_estimator_rate_matrix_path=get_equ_path(),  # This doesn't matter because we are using GT Trees
            #         num_iterations=1,  # This is not needed for coevolution.
            num_processes=num_processes,
            quantization_grid_center=quantization_grid_center,  # This is what we iterate over
            quantization_grid_step=quantization_grid_step,  # This is what we iterate over
            quantization_grid_num_steps=quantization_grid_num_steps,  # This is what we iterate over
            learning_rate=learning_rate,
            num_epochs=num_epochs,
            do_adam=do_adam,
            use_cpp_counting_implementation=use_cpp_implementation,  # Fastest!
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

        # TODO: Don't normalize!
        learned_rate_matrix = learned_rate_matrix.to_numpy()
        # learned_rate_matrix = normalized(learned_rate_matrix.to_numpy())
        # learned_rate_matrix *= 2
        Qs.append(learned_rate_matrix)

        lg_x_lg = read_rate_matrix(
            get_lg_x_lg_path()
        ).to_numpy()  # GT rate matrix
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
    yticks = [np.log(10**i) for i in range(-5, 2)]
    ytickslabels = [f"$10^{{{i + 2}}}$" for i in range(-5, 2)]
    plt.grid()
    plt.ylabel("relative error")
    plt.yticks(yticks, ytickslabels)
    for i, ys in enumerate(yss_relative_errors):
        ys = np.array(ys)
        label = "{:.1f}%".format(100 * np.median(ys))
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
            color="black",
            fontsize=12,
        )  # horizontal alignment can be left, right or center
    plt.title(
        "Distribution of relative error as quantization improves\n(median also reported)"
    )
    plt.tight_layout()
    plt.savefig(f"{output_image_dir}/violin_plot", dpi=300)
    plt.close()


def fig_single_site_cherry_vs_edge():
    for edge_or_cherry in ["edge", "cherry"]:
        output_image_dir = (
            f"images/fig_single_site_cherry_vs_edge/{edge_or_cherry}"
        )
        if not os.path.exists(output_image_dir):
            os.makedirs(output_image_dir)

        num_processes = 32  # Leave some processes open for other work.
        num_sequences = 1024  # To match real data
        num_rate_categories = (
            20  # Only used for creating the GT trees with FastTree.
        )

        # TODO: For co-evolution, we may need more num_families_train?
        num_families_train = None  # We iterate over this
        num_families_test = 0  # We just evaluate l_infty_norm and rmse, we don't look at held out likelihood (we are less interested in held out since we know the ground truth rate matrix)
        ##### To use just the 1448 families with ~210 sites:
        min_num_sites = 190  # We make sure that our simulated MSAs all have roughly the same number of sites, and equal to the median number of sites over the whole dataset (which is 210)
        max_num_sites = 230  # We make sure that our simulated MSAs all have roughly the same number of sites, and equal to the median number of sites over the whole dataset (which is 210)
        min_num_sequences = num_sequences  # We only select families with at least 1024 sequences, such that all the simulated MSAs have exactly the same size. (Even though we don't explore this dimension here, in other experiments we do, and it is nice to be able to compare the results in this figure to those of the other figures, so we use the same training sets across figures if possible.)
        max_num_sequences = 1000000  # We don't want to filter families with more that 1024 sequences, since they will be subsapled later down to 1024.
        ##### To use all 15051 families:
        # min_num_sites = 0  # We make sure that our simulated MSAs all have roughly the same number of sites, and equal to the median number of sites over the whole dataset (which is 210)
        # max_num_sites = 1000000  # We make sure that our simulated MSAs all have roughly the same number of sites, and equal to the median number of sites over the whole dataset (which is 210)
        # min_num_sequences = 0  # We only select families with at least 1024 sequences, such that all the simulated MSAs have exactly the same size. (Even though we don't explore this dimension here, in other experiments we do, and it is nice to be able to compare the results in this figure to those of the other figures, so we use the same training sets across figures if possible.)
        # max_num_sequences = 1000000  # We don't want to filter families with more that 1024 sequences, since they will be subsapled later down to 1024.

        quantization_grid_center = 0.06  # This we will iterate over
        quantization_grid_step = 1.1  # This we will iterate over
        quantization_grid_num_steps = 50  # This we will iterate over
        random_seed = 0  # We fix this, since the number of sequences and sites is large enough to ensure convergence of variance to 0. I.e. there is no variance in this experiment, so we can just set the random seed to 0.
        learning_rate = 3e-2  # We use a highly precise optimizer for this
        num_epochs = 2000  # We use a highly precise optimizer for this
        do_adam = True  # We use a highly precise optimizer for this
        use_cpp_implementation = (
            True  # For simulating MSAs and counting, super fast.
        )

        caching.set_cache_dir("_cache_benchmarking")
        caching.set_hash_len(64)

        num_families_train_list = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]

        yss_relative_errors = []
        Qs = []
        for (i, num_families_train) in enumerate(num_families_train_list):
            msg = f"***** num_families_train = {num_families_train} *****"
            print("*" * len(msg))
            print(msg)
            print("*" * len(msg))

            families_all = get_families_within_cutoff(
                pfam_15k_msa_dir=PFAM_15K_MSA_DIR,  # We use the PFAM 15K dataset as reference for the simulated MSA sizes
                min_num_sites=min_num_sites,  # We make sure that our simulated MSAs all have roughly the same number of sites, and equal to the median number of sites over the whole dataset (which is 210)
                max_num_sites=max_num_sites,  # We make sure that our simulated MSAs all have roughly the same number of sites, and equal to the median number of sites over the whole dataset (which is 210)
                min_num_sequences=min_num_sequences,  # We only select families with at least 1024 sequences, such that all the simulated MSAs have exactly the same size. (Even though we don't explore this dimension here, in other experiments we do, and it is nice to be able to compare the results in this figure to those of the other figures, so we use the same training sets across figures if possible.)
                max_num_sequences=max_num_sequences,  # We don't want to filter families with more that 1024 sequences, since they will be subsapled later down to 1024.
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
                pfam_15k_msa_dir=PFAM_15K_MSA_DIR,  # We use the PFAM 15K as the reference to build the GT trees.
                num_sequences=num_sequences,  # 1024 To match real data
                families=families_all,
                num_rate_categories=num_rate_categories,  # 20, since this is the standard number of rate categories to use.
                num_processes=num_processes,
                random_seed=random_seed,  # We might need to iterate over this.
                use_cpp_simulation_implementation=use_cpp_implementation,  # Fastest!
            )

            # Now run the cherry and oracle edge methods.
            print(f"**** edge_or_cherry = {edge_or_cherry} *****")
            cherry_estimator_res = cherry_estimator(
                msa_dir=msa_dir
                if edge_or_cherry == "cherry"
                else gt_msa_dir,  # Simulated MSAs
                families=families_train,  # We iterate over this
                tree_estimator=partial(  # We use the GT tree estimator
                    gt_tree_estimator,
                    gt_tree_dir=gt_tree_dir,
                    gt_site_rates_dir=gt_site_rates_dir,
                    gt_likelihood_dir=gt_likelihood_dir,
                    num_rate_categories=num_rate_categories,  # This doesn't matter because we are using GT Trees
                ),
                initial_tree_estimator_rate_matrix_path=get_equ_path(),  # This doesn't matter because we are using GT Trees
                num_iterations=1,  # We use GT trees, so no need to iterate.
                num_processes=num_processes,
                quantization_grid_center=quantization_grid_center,  # Fixed
                quantization_grid_step=quantization_grid_step,  # Fixed
                quantization_grid_num_steps=quantization_grid_num_steps,  # Fixed
                learning_rate=learning_rate,
                num_epochs=num_epochs,
                do_adam=do_adam,
                edge_or_cherry=edge_or_cherry,
                use_cpp_counting_implementation=use_cpp_implementation,  # Fastest!
            )

            learned_rate_matrix_path = cherry_estimator_res[
                "learned_rate_matrix_path"
            ]
            learned_rate_matrix = read_rate_matrix(learned_rate_matrix_path)
            # Do not normalize
            #     learned_rate_matrix = normalized(learned_rate_matrix.to_numpy())
            learned_rate_matrix = learned_rate_matrix.to_numpy()

            lg = read_rate_matrix(get_lg_path()).to_numpy()  # GT rate matrix
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

            # TODO: Don't normalize!
            learned_rate_matrix = learned_rate_matrix.to_numpy()
            # learned_rate_matrix = normalized(learned_rate_matrix.to_numpy())
            Qs.append(learned_rate_matrix)

            lg = read_rate_matrix(get_lg_path()).to_numpy()  # GT rate matrix

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
        yticks = [np.log(10**i) for i in range(-5, 2)]
        ytickslabels = [f"$10^{{{i + 2}}}$" for i in range(-5, 2)]
        plt.grid()
        plt.ylabel("relative error")
        plt.yticks(yticks, ytickslabels)
        for i, ys in enumerate(yss_relative_errors):
            ys = np.array(ys)
            label = "{:.1f}%".format(100 * np.median(ys))
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
                color="black",
                fontsize=12,
            )  # horizontal alignment can be left, right or center
        plt.title(
            "Distribution of relative error as quantization improves\n(median also reported)"
        )
        plt.tight_layout()
        plt.savefig(f"{output_image_dir}/violin_plot", dpi=300)
        plt.close()


def fig_pair_site_number_of_families():
    output_image_dir = "images/fig_pair_site_number_of_families"
    if not os.path.exists(output_image_dir):
        os.makedirs(output_image_dir)

    num_processes = 32  # Leave some processes open for other work.
    num_sequences = (
        1024  # We use a lot of sequences per family to take variance to 0.
    )
    num_rate_categories = (
        20  # Only used for craeting the GT trees with FastTree.
    )

    num_families_train = None  # We iterate over this
    num_families_test = 0  # We just evaluate l_infty_norm and rmse, we don't look at held out likelihood (we are less interested in held out since we know the ground truth rate matrix)

    quantization_grid_center = 0.06  # This we will iterate over
    quantization_grid_step = 1.1  # This we will iterate over
    quantization_grid_num_steps = 50  # This we will iterate over
    random_seed = 0  # We fix this, since the number of sequences and sites is large enough to ensure convergence of variance to 0. I.e. there is no variance in this experiment, so we can just set the random seed to 0.
    learning_rate = 3e-2  # 3 * (1e-2)
    do_adam = True
    use_cpp_implementation = (
        True  # For simulating MSAs and counting, super fast.
    )
    minimum_distance_for_nontrivial_contact = (
        7  # We use the standard of 7 position away.
    )
    num_epochs = 200  # An accurate optimizer (for single-site model). Only 200 epochs since slower.
    angstrom_cutoff = 8.0  # Standard cutoff for determining contacts.

    caching.set_cache_dir("_cache_benchmarking")
    caching.set_hash_len(64)

    num_families_train_list = [
        1024,
        2048,
        4096,
        8192,
    ]  # [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 15051]:
    yss_relative_errors = []
    Qs = []
    for (i, num_families_train) in enumerate(num_families_train_list):
        msg = f"***** num_families_train = {num_families_train} *****"
        print("*" * len(msg))
        print(msg)
        print("*" * len(msg))

        if num_families_train <= 1024:
            families_all = get_families_within_cutoff(
                pfam_15k_msa_dir=PFAM_15K_MSA_DIR,  # We use the PFAM 15K dataset as reference for the simulated MSA sizes
                min_num_sites=190,  # We make sure that our simulated MSAs all have roughly the same number of sites, and equal to the median number of sites over the whole dataset (which is 210)
                max_num_sites=230,  # We make sure that our simulated MSAs all have roughly the same number of sites, and equal to the median number of sites over the whole dataset (which is 210)
                min_num_sequences=num_sequences,  # We only select families with at least 1024 sequences, such that all the simulated MSAs have exactly the same size. (Even though we don't explore this dimension here, in other experiments we do, and it is nice to be able to compare the results in this figure to those of the other figures, so we use the same training sets across figures if possible.)
                max_num_sequences=1000000,  # We don't want to filter families with more that 1024 sequences, since they will be subsapled later down to 1024.
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
            pfam_15k_msa_dir=PFAM_15K_MSA_DIR,  # We use the PFAM 15K as the reference to build the GT trees.
            pfam_15k_pdb_dir=PFAM_15K_PDB_DIR,
            minimum_distance_for_nontrivial_contact=minimum_distance_for_nontrivial_contact,
            angstrom_cutoff=angstrom_cutoff,
            num_sequences=num_sequences,  # 1024
            families=families_all,
            num_rate_categories=num_rate_categories,  # 20, since this is the standard number of rate categories to use.
            num_processes=num_processes,
            random_seed=random_seed,  # This will be fixed to 0, since the dataset is large, making variance vanish and bias show itself.
            use_cpp_simulation_implementation=use_cpp_implementation,  # Fastest!
        )

        # Now run the cherry method with access to GT trees
        cherry_estimator_res = cherry_estimator_coevolution(
            msa_dir=msa_dir,  # Simulated MSAs
            contact_map_dir=contact_map_dir,  # Synthetic contact maps.
            minimum_distance_for_nontrivial_contact=minimum_distance_for_nontrivial_contact,
            coevolution_mask_path="data/mask_matrices/aa_coevolution_mask.txt",
            families=families_train,  # 1024
            tree_estimator=partial(  # We use the GT tree estimator
                gt_tree_estimator,
                gt_tree_dir=gt_tree_dir,
                gt_site_rates_dir=gt_site_rates_dir,
                gt_likelihood_dir=gt_likelihood_dir,
                num_rate_categories=num_rate_categories,  # This doesn't matter because we are using GT Trees
            ),
            initial_tree_estimator_rate_matrix_path=get_equ_path(),  # This doesn't matter because we are using GT Trees
            #         num_iterations=1,  # This is not needed for coevolution.
            num_processes=num_processes,
            quantization_grid_center=quantization_grid_center,  # This is what we iterate over
            quantization_grid_step=quantization_grid_step,  # This is what we iterate over
            quantization_grid_num_steps=quantization_grid_num_steps,  # This is what we iterate over
            learning_rate=learning_rate,
            num_epochs=num_epochs,
            do_adam=do_adam,
            use_cpp_counting_implementation=use_cpp_implementation,  # Fastest!
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

        # TODO: Don't normalize!
        learned_rate_matrix = learned_rate_matrix.to_numpy()
        # learned_rate_matrix = normalized(learned_rate_matrix.to_numpy())
        # learned_rate_matrix *= 2
        Qs.append(learned_rate_matrix)

        lg_x_lg = read_rate_matrix(
            get_lg_x_lg_path()
        ).to_numpy()  # GT rate matrix
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
    yticks = [np.log(10**i) for i in range(-5, 2)]
    ytickslabels = [f"$10^{{{i + 2}}}$" for i in range(-5, 2)]
    plt.grid()
    plt.ylabel("relative error")
    plt.yticks(yticks, ytickslabels)
    for i, ys in enumerate(yss_relative_errors):
        ys = np.array(ys)
        label = "{:.1f}%".format(100 * np.median(ys))
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
            color="black",
            fontsize=12,
        )  # horizontal alignment can be left, right or center
    plt.title(
        "Distribution of relative error as quantization improves\n(median also reported)"
    )
    plt.tight_layout()
    plt.savefig(f"{output_image_dir}/violin_plot", dpi=300)
    plt.close()
    print("Done!")
