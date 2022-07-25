from figures import fig_single_site_quantization_error, fig_pair_site_quantization_error, fig_single_site_cherry_vs_edge, live_demo_single_site, live_demo_pair_of_sites, fig_lg_paper, fig_single_site_learning_rate_robustness, fig_convergence_on_infinite_data_single_site, fig_convergence_on_large_data_single_site, fig_convergence_on_infinite_data_pair_site, fig_convergence_on_large_data_pair_site, fig_jtt_ipw_single_site, fig_pfam15k, fig_convergence_on_large_data_single_site__variance, fig_single_site_em

if __name__ == "__main__":
    print("Main starting ...")

    # live_demo_pair_of_sites()
    # fig_lg_paper()
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
    # # live_demo_single_site()
    # fig_jtt_ipw_single_site(use_best_iterate=True)
    # fig_jtt_ipw_single_site(use_best_iterate=False)

    # fig_convergence_on_infinite_data_pair_site(use_best_iterate=True)
    # fig_convergence_on_infinite_data_pair_site(use_best_iterate=False)
    # fig_convergence_on_large_data_pair_site(use_best_iterate=True)
    # fig_convergence_on_large_data_pair_site(use_best_iterate=False)
    # fig_pair_site_quantization_error(use_best_iterate=True)
    # fig_pair_site_quantization_error(use_best_iterate=False)

    # fig_pfam15k(num_rate_categories=1)
    # fig_pfam15k(num_rate_categories=2)
    # fig_pfam15k(num_rate_categories=4)
    # fig_pfam15k(num_rate_categories=20)

    # fig_qmaker()
    # from src.phylogeny_estimation._iq_tree import _install_iq_tree
    # _install_iq_tree()

    # fig_single_site_em("-band 0 -fixgaprates -mininc 0.001 -maxiter 10000 -nolaplace")
    # fig_single_site_em("-band 0 -fixgaprates -mininc 0.0001 -maxiter 10000 -nolaplace")
    # fig_single_site_em("-band 0 -fixgaprates -mininc 0.00001 -maxiter 10000 -nolaplace")

    print("Main done!"