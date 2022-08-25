from figures import fig_single_site_quantization_error, fig_pair_site_quantization_error, fig_single_site_cherry_vs_edge, live_demo_single_site, live_demo_pair_of_sites, fig_lg_paper, fig_single_site_learning_rate_robustness, fig_convergence_on_infinite_data_single_site, fig_convergence_on_large_data_single_site, fig_convergence_on_infinite_data_pair_site, fig_convergence_on_large_data_pair_site, fig_jtt_ipw_single_site, fig_pfam15k, fig_convergence_on_large_data_single_site__variance, fig_single_site_em, fig_single_site_cherry_vs_edge_num_sequences, fig_MSA_VI_cotransition

if __name__ == "__main__":
    print("Main starting ...")

    # fig_lg_paper(
    #     figsize=(6.4, 4.8),
    #     show_legend=True,
    # )  # Note: Cannot run Historian because it models gap instead of treating them as MACAR...

    # fig_single_site_quantization_error(
    #     use_best_iterate=True,
    #     num_rate_categories=4
    # )
    # fig_single_site_cherry_vs_edge(
    #     use_best_iterate=True,
    #     num_rate_categories=4
    # )
    # fig_pfam15k(
    #     num_rate_categories=1,
    #     num_families_train=15051,
    #     num_families_test=1,
    # )
    # # TODO: Will need to point these to the new rate matrices estimated by fig_pfam15k
    # fig_pair_site_quantization_error(use_best_iterate=True, Q_2_name="unmasked-co-transitions", num_rate_categories=1)  # Works!
    # fig_pair_site_quantization_error(use_best_iterate=True, Q_2_name="unmasked-single-transitions", num_rate_categories=1)  # Works!
    # fig_single_site_em(
    #     "-band 0 -fixgaprates -mininc 0.000001 -maxiter 100000000 -nolaplace",
    #     num_processes=32,
    #     num_rate_categories=4
    # ) # DONE: Optimal

    # fig_MSA_VI_cotransition(
    #     num_families_train=10,
    #     aa_1="K",
    #     aa_2="E",
    #     families = ["4kv7_1_A"],
    # )
    print("Main done!")
