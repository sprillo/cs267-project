import os
import time

from src.evaluation import compute_log_likelihoods
from src.caching import set_cache_dir, set_hash
from src.io import write_probability_distribution, write_rate_matrix
from src.utils import amino_acids
from src.markov_chain import wag_matrix, compute_stationary_distribution, chain_product

TREE_DIR = "/export/home/users/sprillo/Git/Phylo-correction/cs267_data/trees_1024_seqs_None_sites_LG_FastTree.txt-d15ceeb4_RM_20_cats"
MSA_DIR = "/export/home/users/sprillo/Git/Phylo-correction/cs267_data/msas_1024_seqs_None_sites_LG_FastTree.txt-d15ceeb4_RM_20_cats"
SITE_RATES_DIR = "/export/home/users/sprillo/Git/Phylo-correction/cs267_data/site_rates_1024_seqs_None_sites_LG_FastTree.txt-d15ceeb4_RM_20_cats"
CONTACT_MAP_DIR = "/export/home/users/sprillo/Git/Phylo-correction/cs267_data/contact_maps_1024_seqs_None_sites_LG_FastTree.txt-d15ceeb4_RM_20_cats_maximal_matching"
filenames = sorted(list(os.listdir("/export/home/users/sprillo/Git/Phylo-correction/cs267_data/contact_maps_1024_seqs_None_sites_LG_FastTree.txt-d15ceeb4_RM_20_cats")))
protein_family_names = [x.split(".")[0] for x in filenames]
RATE_MATRICES_PATH = "data/rate_matrices"
PI_1_PATH = os.path.join(RATE_MATRICES_PATH, "wag_stationary.txt")
Q_1_PATH = os.path.join(RATE_MATRICES_PATH, "wag.txt")
PI_2_PATH = os.path.join(RATE_MATRICES_PATH, "wag_x_wag_stationary.txt")
Q_2_PATH = os.path.join(RATE_MATRICES_PATH, "wag_x_wag.txt")

if not os.path.exists(PI_1_PATH):
    pairs_of_amino_acids = [aa1 + aa2 for aa1 in amino_acids for aa2 in amino_acids]
    wag = wag_matrix().to_numpy()
    write_rate_matrix(wag, amino_acids, Q_1_PATH)
    pi = compute_stationary_distribution(wag)
    write_probability_distribution(pi, amino_acids, PI_1_PATH)
    wag_x_wag = chain_product(wag, wag)
    write_rate_matrix(wag_x_wag, pairs_of_amino_acids, Q_2_PATH)
    pi_x_pi = compute_stationary_distribution(wag_x_wag)
    write_probability_distribution(pi_x_pi, pairs_of_amino_acids, PI_2_PATH)

compute_log_likelihoods(
    tree_dir=TREE_DIR,
    msa_dir=MSA_DIR,
    site_rates_dir=SITE_RATES_DIR,
    contact_map_dir=CONTACT_MAP_DIR,
    amino_acids=amino_acids,
    families=protein_family_names[:1],
    pi_1_path=PI_1_PATH,
    Q_1_path=Q_1_PATH,
    pi_2_path=PI_2_PATH,
    Q_2_path=Q_2_PATH,
    reversible_1=True,
    device_1='cpu',
    reversible_2=True,
    device_2='cpu',
    output_likelihood_dir='./test_outputs',
    num_processes=1,
    use_cpp_implementation=False,
)
