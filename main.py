from figures import fig_single_site_quantization_error, fig_pair_site_quantization_error, fig_single_site_cherry_vs_edge, live_demo_single_site, live_demo_pair_of_sites, fig_lg_paper, fig_single_site_learning_rate_robustness, fig_convergence_on_infinite_data_single_site, fig_convergence_on_large_data_single_site, fig_convergence_on_infinite_data_pair_site, fig_convergence_on_large_data_pair_site, fig_jtt_ipw_single_site, fig_pfam15k, fig_convergence_on_large_data_single_site__variance, fig_single_site_em, fig_single_site_cherry_vs_edge_num_sequences, fig_MSA_VI_cotransition

if __name__ == "__main__":
    print("Main starting ...")

    # live_demo_pair_of_sites()
    # fig_lg_paper()  # Note: Cannot run Historian because it models gap instead of treating them as MACAR...
    # fig_single_site_learning_rate_robustness()

    # debug_pytorch_optimizer()

    # fig_convergence_on_infinite_data_single_site(use_best_iterate=True)
    # fig_convergence_on_infinite_data_single_site(use_best_iterate=False)
    # fig_convergence_on_large_data_single_site__variance(use_best_iterate=True)
    # fig_convergence_on_large_data_single_site__variance(use_best_iterate=False)
    # fig_convergence_on_large_data_single_site(use_best_iterate=True)
    # fig_convergence_on_large_data_single_site(use_best_iterate=False)
    # fig_single_site_quantization_error(use_best_iterate=True)
    # fig_single_site_quantization_error(use_best_iterate=False)
    # fig_single_site_cherry_vs_edge(use_best_iterate=True)
    # fig_single_site_cherry_vs_edge(use_best_iterate=False)
    # fig_single_site_cherry_vs_edge_num_sequences(use_best_iterate=True)  # TODO: Check results!
    # fig_single_site_cherry_vs_edge_num_sequences(use_best_iterate=False)
    # # live_demo_single_site()
    # fig_jtt_ipw_single_site(use_best_iterate=True)
    # fig_jtt_ipw_single_site(use_best_iterate=False)

    # fig_convergence_on_infinite_data_pair_site(use_best_iterate=True)
    # fig_convergence_on_infinite_data_pair_site(use_best_iterate=False)
    # fig_convergence_on_large_data_pair_site(use_best_iterate=True)
    # fig_convergence_on_large_data_pair_site(use_best_iterate=False)
    # fig_pair_site_quantization_error(use_best_iterate=True, Q_2_name="lg_x_lg")  # DONE: Make sure it repros the Adobe Illustrator plot.
    # fig_pair_site_quantization_error(use_best_iterate=True, Q_2_name="masked")  # TODO: Run on full range of quantization values.
    # fig_pair_site_quantization_error(use_best_iterate=True, Q_2_name="unmasked-all-transitions")  # Works!
    # fig_pair_site_quantization_error(use_best_iterate=True, Q_2_name="unmasked-co-transitions")  # Works!
    # fig_pair_site_quantization_error(use_best_iterate=True, Q_2_name="unmasked-single-transitions")  # Works!
    # fig_pair_site_quantization_error(use_best_iterate=False)

    # fig_pfam15k(num_rate_categories=1)
    # fig_pfam15k(num_rate_categories=2)
    # fig_pfam15k(num_rate_categories=4)
    # fig_pfam15k(num_rate_categories=20)

    # fig_qmaker()
    # from src.phylogeny_estimation._iq_tree import _install_iq_tree
    # _install_iq_tree()

    # fig_single_site_em(
    #     "-band 0 -fixgaprates -mininc 0.001 -maxiter 10000 -nolaplace",
    #     num_processes=4,
    # )
    # fig_single_site_em(
    #     "-band 0 -fixgaprates -mininc 0.0001 -maxiter 10000 -nolaplace",
    #     num_processes=4,
    # )
    # fig_single_site_em(
    #     "-band 0 -fixgaprates -mininc 0.00001 -maxiter 10000 -nolaplace",
    #     num_processes=4,
    # ) # DONE: Worse than 0.000001
    # fig_single_site_em(
    #     "-band 0 -fixgaprates -mininc 0.000001 -maxiter 10000 -nolaplace",
    #     num_processes=4,
    # ) # DONE: Optimal
    # fig_single_site_em(
    #     "-band 0 -fixgaprates -mininc 0.0000001 -maxiter 10000 -nolaplace",
    #     num_processes=4,
    # )  # DONE: Not much better than 0.000001, and takes twice as long

    fig_MSA_VI_cotransition(num_families_train=10)

    print("Main done!")
