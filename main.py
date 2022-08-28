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
    fig_single_site_cherry_vs_edge,
    fig_single_site_cherry_vs_edge_num_sequences,
    fig_single_site_em,
    fig_single_site_learning_rate_robustness,
    fig_single_site_quantization_error,
    learn_coevolution_model_on_pfam15k,
    live_demo_pair_of_sites,
    live_demo_single_site,
)

if __name__ == "__main__":
    print("Main starting ...")

    # fig_lg_paper()
    # fig_single_site_quantization_error()
    # fig_single_site_cherry_vs_edge()
    # learn_coevolution_model_on_pfam15k()
    # TODO: Will need to point these to the new rate matrices estimated by fig_pfam15k
    # fig_pair_site_quantization_error(
    #     Q_2_name="unmasked-co-transitions",
    # )
    # fig_pair_site_quantization_error(
    #     Q_2_name="unmasked-single-transitions",
    # )
    # fig_single_site_em()

    # fig_MSA_VI_cotransition(
    #     num_families_train=10,
    #     aa_1="K",
    #     aa_2="E",
    #     families = ["4kv7_1_A"],
    # )
    print("Main done!")
