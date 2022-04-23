from typing import List

import logging

# PhylogenyGenerator(
#     a3m_dir_full=a3m_dir_full,
#     a3m_dir=a3m_dir,
#     n_process=n_process,
#     expected_number_of_MSAs=expected_number_of_MSAs,
#     outdir=tree_dir,
#     max_seqs=max_seqs,
#     max_sites=max_sites,
#     max_families=max_families,
#     rate_matrix=rate_matrix,
#     fast_tree_cats=fast_tree_cats,
#     use_cached=use_cached,
# )

# @cached_parallel_computation(
#     exclude_args=["num_processes"],
#     parallel_arg="families",
#     output_dirs=["output_tree_dir"],
# )
def fast_tree(
    msa_dir: str,
    families: List[str],
    num_processes: int,
    output_tree_dir: str,
    rate_matrix_path: str,
    num_rate_categories: int,
) -> None:
    logger = logging.getLogger("rate_estimation.fast_tree")

    dir_path = os.path.dirname(os.path.realpath(_file_))
    c_path = os.path.join(dir_path, 'FastTree.c')
    bin_path = os.path.join(dir_path, 'FastTree')
    if not os.path.exists(bin_path):
        os.system(
            f"wget http://www.microbesonline.org/fasttree/FastTree.c -P {dir_path}"
        )
        compile_command = f"gcc -DNO_SSE -DUSE_DOUBLE -O3 -finline-functions -funroll-loops -Wall -o {bin_path} {c_path} -lm"
        logger.info(f"Compiling FastTree with:\n{compile_command}")
        # See http://www.microbesonline.org/fasttree/#Install
        os.system(compile_command)
    raise NotImplementedError
