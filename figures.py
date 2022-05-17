from src.benchmarking.pfam_15k import simulate_ground_truth_data_single_site, get_families_within_cutoff
from src.markov_chain import get_equ_path, normalized, get_lg_path
from src import cherry_estimator
from src import caching
from src.evaluation import l_infty_norm, rmse, mre, mean_relative_error, relative_errors, plot_rate_matrix_predictions
from src.io import read_rate_matrix
from src.phylogeny_estimation import gt_tree_estimator
from functools import partial
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

PFAM_15K_MSA_DIR = "input_data/a3m"
PFAM_15K_PDB_DIR = "input_data/pdb"


def fig_single_site_quantization_error():
    output_image_dir = "images/fig_single_site_quantization_error"
    if not os.path.exists(output_image_dir):
        os.makedirs(output_image_dir)

    num_processes = 32  # Leave some processes open for other work.
    num_sequences = 1024  # We use a lot of sequences per family to take variance to 0.
    num_rate_categories = 20  # Only used for creating the GT trees with FastTree.
    num_families_train = 1024  # We fix the number of families to a large number of 1024, to take variance to 0.
    num_families_test = 0  # We just evaluate l_infty_norm and rmse, we don't look at held out likelihood (we are less interested in held out since we know the ground truth rate matrix)
    quantization_grid_center = None  # This we will iterate over
    quantization_grid_step = None  # This we will iterate over
    quantization_grid_num_steps = None  # This we will iterate over
    random_seed = 0  # We fix this, since the number of sequences and sites is large enough to ensure convergence of variance to 0. I.e. there is no variance in this experiment, so we can just set the random seed to 0.
    learning_rate = 3e-2  # We use a highly precise optimizer for this
    num_epochs = 2000  # We use a highly precise optimizer for this
    do_adam = True  # We use a highly precise optimizer for this
    use_cpp_implementation = True  # For simulating MSAs and counting, super fast.

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
    ys_linfty = []
    ys_rmse = []
    ys_mean_relative_error = []
    yss_relative_errors = []
    Qs = []
    for (quantization_grid_center, quantization_grid_step, quantization_grid_num_steps) in qs:
        msg = f"***** grid = {(quantization_grid_center, quantization_grid_step, quantization_grid_num_steps)} *****"
        print("*" * len(msg))
        print(msg)
        print("*" * len(msg))

        families_all = get_families_within_cutoff(
            pfam_15k_msa_dir=PFAM_15K_MSA_DIR,  # We use the PFAM 15K dataset as reference for the simulated MSA sizes
            min_num_sites=190,  # We make sure that our simulated MSAs all have roughly the same number of sites, and equal to the median number of sites over the whole dataset (which is 210)
            max_num_sites=230,  # We make sure that our simulated MSAs all have roughly the same number of sites, and equal to the median number of sites over the whole dataset (which is 210)
            min_num_sequences=num_sequences,  # We only select families with at least 1024 sequences, such that all the simulated MSAs have exactly the same size. (Even though we don't explore this dimension here, in other experiments we do, and it is nice to be able to compare the results in this figure to those of the other figures, so we use the same training sets across figures if possible.)
            max_num_sequences=1000000,  # We don't want to filter families with more that 1024 sequences, since they will be subsapled later down to 1024.
        )
        families_train = families_all[:num_families_train]
        if num_families_test == 0:
            families_test = []
        else:
            families_test = families_all[-num_families_test:]
        print(f"len(families_all) = {len(families_all)}")
        if num_families_train + num_families_test > len(families_all):
            raise Exception(f"Training and testing set would overlap!")
        assert(len(set(families_train + families_test)) == len(families_train) + len(families_test))

        msa_dir, contact_map_dir, gt_msa_dir, gt_tree_dir, gt_site_rates_dir, gt_likelihood_dir = simulate_ground_truth_data_single_site(
            pfam_15k_msa_dir=PFAM_15K_MSA_DIR,  # We use the PFAM 15K as the reference to build the GT trees.
            num_sequences=num_sequences,  # 1024
            families=families_train + families_test,  # 1024 families
            num_rate_categories=num_rate_categories,  # 20, since this is the standard number of rate categories to use.
            num_processes=num_processes,
            random_seed=random_seed,  # This will be fixed to 0, since the dataset is large, making variance vanish and bias show itself.
            use_cpp_simulation_implementation=use_cpp_implementation, # Fastest!
        )

        # Now run the cherry method with access to GT trees
        learned_rate_matrix_path = cherry_estimator(
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
        learned_rate_matrix = read_rate_matrix(
            learned_rate_matrix_path
        )
        learned_rate_matrix = normalized(learned_rate_matrix.to_numpy())
        Qs.append(learned_rate_matrix)

        lg = read_rate_matrix(get_lg_path()).to_numpy()  # GT rate matrix

        yss_relative_errors.append(
            relative_errors(
                lg,
                learned_rate_matrix
            )
        )

    for i in range(len(q_points)):
        plot_rate_matrix_predictions(
            read_rate_matrix(get_lg_path()).to_numpy(),
            Qs[i]
        )
        plt.title(f"True vs predicted rate matrix entries\nmax quantization error = %.1f%% (%i quantization points)" % (q_errors[i], q_points[i]))
        plt.savefig(f"{output_image_dir}/result_{i}", dpi=300)
        plt.close()

    df = pd.DataFrame(
        {
            "quantization points": sum([[q_points[i]] * len(yss_relative_errors[i]) for i in range(len(yss_relative_errors))], []),
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
    yticks = [np.log(10 ** i) for i in range(-5, 2)]
    ytickslabels = [f"$10^{{{i + 2}}}$%" for i in range(-5, 2)]
    plt.grid()
    plt.ylabel("relative error")
    plt.yticks(yticks, ytickslabels)
    for i, ys in enumerate(yss_relative_errors):
        ys = np.array(ys)
        label = "{:.1f}%".format(100 * np.median(ys))
        plt.annotate(label, # this is the text
                    (i + 0.05, np.log(np.max(ys))), # these are the coordinates to position the label
                    textcoords="offset points", # how to position the text
                    xytext=(0, 10), # distance from text to points (x,y)
                    ha='left',
                    va='top',
                    color='black',
                    fontsize=12
                    ) # horizontal alignment can be left, right or center
    plt.title("Distribution of relative error as quantization improves\n(median also reported)")
    plt.tight_layout()
    plt.savefig(f"{output_image_dir}/result", dpi=300)
    plt.close()
