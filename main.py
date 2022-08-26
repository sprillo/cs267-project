from figures import (
    fig_convergence_on_infinite_data_pair_site,
    fig_convergence_on_infinite_data_single_site,
    fig_convergence_on_large_data_pair_site,
    fig_convergence_on_large_data_single_site,
    fig_convergence_on_large_data_single_site__variance,
    fig_jtt_ipw_single_site,
    fig_lg_paper,
    fig_MSA_VI_cotransition,
    fig_pair_site_quantization_error,
    fig_pfam15k,
    fig_single_site_cherry_vs_edge,
    fig_single_site_cherry_vs_edge_num_sequences,
    fig_single_site_em,
    fig_single_site_learning_rate_robustness,
    fig_single_site_quantization_error,
    live_demo_pair_of_sites,
    live_demo_single_site,
)

if __name__ == "__main__":
    print("Main starting ...")

    # fig_lg_paper()

    # fig_single_site_quantization_error()
    # fig_single_site_cherry_vs_edge()
    # fig_pfam15k(
    #     num_rate_categories=1,
    #     num_families_train=15051,
    #     num_families_test=1,
    #     num_processes=8,
    # )
    # TODO: Will need to point these to the new rate matrices estimated by fig_pfam15k
    # fig_pair_site_quantization_error(
    #     use_best_iterate=True,
    #     Q_2_name="unmasked-co-transitions",
    #     num_rate_categories=1,
    #     num_processes=8,
    # )  # Works!
    # fig_pair_site_quantization_error(
    #     use_best_iterate=True,
    #     Q_2_name="unmasked-single-transitions",
    #     num_rate_categories=1,
    #     num_processes=8,
    # )  # Works!
    # fig_single_site_em(
    #     "-band 0 -fixgaprates -mininc 0.000001 -maxiter 100000000 -nolaplace",
    #     num_processes=4,
    #     num_rate_categories=4
    # ) # DONE: Optimal

    # fig_MSA_VI_cotransition(
    #     num_families_train=10,
    #     aa_1="K",
    #     aa_2="E",
    #     families = ["4kv7_1_A"],
    # )
    print("Main done!")
