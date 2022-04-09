# CS267 Project: Scalable Estimation and Evaluation of Models of Amino-Acid Co-Evolution

This repo contains starter code for our CS267 class project. The starter code consists of parallel Python implementations and automated correctness tests.

## Getting set up

To get set up on Cori:
```
module load python/3.8-anaconda-2020.11
```

Then install requirements:
```
pip install -r requirements.txt
```

Then run all the fast tests (takes ~ 10 seconds):
```
python -m pytest tests
```

If the tests pass, you are good to go.

## Running tests for a specific module

To run ONLY the counting fast tests (takes ~ 1 second):
```
python -m pytest tests/counting_tests/
```

To run ONLY the counting fast AND SLOW tests (takes ~ 2 minutes, requires 32 cores):
```
python -m pytest tests/counting_tests/ --runslow
```

To run ONLY the simulation fast tests (takes ~ 10 seconds):
```
python -m pytest tests/simulation_tests/
```

There are no simulation slow tests (if you pass the fast tests you are good).

## Main Modules

### src.counting

This module contains the functions to compute count matrices used to estimate rate matrices. The two functions exposed by this module are `count_transitions` and `count_co_transitions`. The function `count_transitions` counts transitions between _single_ amino acids, while the function `count_co_transitions` counts transitions between _pairs_ of amino acids. Our goal is to write an efficient C++ implementation for `count_co_transitions`, which should be called from the Python code when `use_cpp_implementation=True`. Currently, there is no C++ implementation, so it will raise an error:

```
if use_cpp_implementation:
    raise NotImplementedError
```

It might be easier to start with `count_transitions` since it is single-site. The signature of the `count_co_transitions` function is:

```
def count_co_transitions(
    tree_dir: str,
    msa_dir: str,
    contact_map_dir: str,
    families: List[str],
    amino_acids: List[str],
    quantization_points: List[float],
    edge_or_cherry: bool,
    minimum_distance_for_nontrivial_contact: int,
    output_count_matrices_dir: str,
    num_processes: int,
    use_cpp_implementation: bool = False,
) -> None
```

Importantly, here `tree_dir`, `msa_dir`, and `contact_map_dir` refer to the directories containing the trees, msas, and contact maps. Furthermore, `families` specifies for what protein families we should perform the computation. This easily allows testing the method on a subset of the protein families before running it on all of them. `output_count_matrices_dir` is where the count matrices should be writted to, in a file simply called `result.txt`.

Take a look at the docstring of these functions for more details, as well as the tests at `tests/counting_tests/counting_test.py`. You are free to add any C++ specific flags to the Python API (such as number of OpenMP threads, number of nodes, etc.), such that they get forwarded to your C++ program.

### src.simulation

This module exposes a unique function `simulate_msas` which simulates data under a given Markov Chain model of amino acid evolution. Our goal is to write an efficient C++ implementation for `simulate_msas`, which should be called from the Python code when `use_cpp_implementation=True`. Currently, there is no C++ implementation, so it will raise an error:

```
if use_cpp_implementation:
    raise NotImplementedError
```

The signature of the `simulate_msas` function is:

```
def simulate_msas(
    tree_dir: str,
    site_rates_dir: str,
    contact_map_dir: str,
    families: List[str],
    amino_acids: List[str],
    pi_1_path: str,
    Q_1_path: str,
    pi_2_path: str,
    Q_2_path: str,
    strategy: str,
    output_msa_dir: str,
    random_seed: int,
    num_processes: int,
    use_cpp_implementation: bool = False,
) -> None:
```

Importantly, here `tree_dir`, `site_rates_dir`, and `contact_map_dir` refer to the directories containing the trees, site rates, and contact maps. Furthermore, `families` specifies for what protein families we should perform the computation. This easily allows testing the method on a subset of the protein families before running it on all of them. `output_msa_dir` is where the simulated MSAs should be writted to, one per input family, at the file {family_name}.txt.

Take a look at the docstring of the function for more details, as well as the tests at `tests/simulation_tests/simulation_test.py`. You are free to add any C++ specific flags to the Python API (such as number of OpenMP threads, number of nodes, etc.), such that they get forwarded to your C++ program.

### src.evaluation

This module exposes a unique function `compute_log_likelihoods` which computes data log likelihood under a given Markov Chain model of amino acid evolution. This is the counterpart to `simulate_msas`, in the sense that `simulate_msas` can simulate data from the model, while `compute_log_likelihoods` can compute the likelihood of data under the model. Our goal is to write an efficient C++ implementation for `compute_log_likelihoods`, and I will take care of it to begin with (I still have to write the Python code for it).

## Datasets

There are 4 kinds of data files:

- Trees
- Multiple sequence alignments (MSAs)
- Contact maps
- Site rates

Each protein family has exactly one file of each kind associated with it. We store trees, MSAs, contact maps, and site rates in separate directories, such as the tiny test dataset at `tests/counting_tests/test_input_data/tiny`, which contains (among other test data):

- `tree_dir`
- `msa_dir`
- `contact_map_dir`
- `site_rates_dir`

Each protein family contains a file in each of these directories.

The _real_ datasets we will be working with have 15051 protein families and are located under `/global/cscratch1/sd/sprillo/cs267_data`. There are 3 real datasets with increasing sizes: one with 1024 sequences per family, one with 2048, and one with 4096. For example, the dataset with 4096 sequences per family is composed of:

- `trees_4096_seqs_None_sites_LG_FastTree.txt-d15ceeb4_RM_20_cats`
- `msas_4096_seqs_None_sites_LG_FastTree.txt-d15ceeb4_RM_20_cats`
- `contact_maps_4096_seqs_None_sites_LG_FastTree.txt-d15ceeb4_RM_20_cats`
- `site_rates_4096_seqs_None_sites_LG_FastTree.txt-d15ceeb4_RM_20_cats`

You can ignore the suffix `None_sites_LG_FastTree.txt-d15ceeb4_RM_20_cats` - this has to do with how I generated the data. Just look at the prefix, e.g. `trees_4096...`, `msas_4096...` etc. to know what dataset you are using. Here are the sizes of the datasets for you to get a feel:

```
sprillo@cori03: /global/cscratch1/sd/sprillo/cs267_data () $ du -h --max-depth=1 . | sort -h
66M    ./site_rates_1024_seqs_None_sites_LG_FastTree.txt-d15ceeb4_RM_20_cats
66M    ./site_rates_2048_seqs_None_sites_LG_FastTree.txt-d15ceeb4_RM_20_cats
66M    ./site_rates_4096_seqs_None_sites_LG_FastTree.txt-d15ceeb4_RM_20_cats
1.2G   ./trees_1024_seqs_None_sites_LG_FastTree.txt-d15ceeb4_RM_20_cats
1.3G   ./contact_maps_1024_seqs_None_sites_LG_FastTree.txt-d15ceeb4_RM_20_cats
1.3G   ./contact_maps_2048_seqs_None_sites_LG_FastTree.txt-d15ceeb4_RM_20_cats
1.3G   ./contact_maps_4096_seqs_None_sites_LG_FastTree.txt-d15ceeb4_RM_20_cats
2.3G   ./trees_2048_seqs_None_sites_LG_FastTree.txt-d15ceeb4_RM_20_cats
3.6G   ./msas_1024_seqs_None_sites_LG_FastTree.txt-d15ceeb4_RM_20_cats
3.8G   ./trees_4096_seqs_None_sites_LG_FastTree.txt-d15ceeb4_RM_20_cats
6.8G   ./msas_2048_seqs_None_sites_LG_FastTree.txt-d15ceeb4_RM_20_cats
12G    ./msas_4096_seqs_None_sites_LG_FastTree.txt-d15ceeb4_RM_20_cats
33G    .
```

To make your life easier, a subset of 32 families from the dataset with 1024 sequences is commited at `tests/counting_tests/test_input_data/medium`. It is used for testing the counting method in `tests/counting_tests/counting_test.py`, and you can consider using it as you implement your method to e.g. run it locally on your own machine, or get a feeling for how fast things are before you run them on the large datasets. The name of these 32 families are listed in the `families_medium` variable in the `tests/counting_tests/counting_test.py` module.

### Trees

A tree file looks like this (taken from `tests/counting_tests/test_input_data/tiny/tree_dir/fam1.txt`):

```
6 nodes
internal-0
internal-1
internal-2
seq1
seq2
seq3
5 edges
internal-0 internal-1 1.0
internal-1 internal-2 2.0
internal-2 seq1 3.0
internal-2 seq2 4.0
internal-1 seq3 5.0
```

The first line indicates how many nodes are in the tree, followed by the node names, followed by the number of edges in the tree, followed by the edges and their lengths.

### Multiple sequence alignments (MSAs)

An MSA file looks like this (taken from `tests/counting_tests/test_input_data/tiny/tree_dir/fam1.txt`):

```
>internal-0
SSIIS
>internal-1
SSIIS
>internal-2
TTLLS
>seq1
TTLLS
>seq2
TTIIS
>seq3
SSIIS
```

This encodes a dictionary `{"internal-0": "SSIIS", "internal-1": "SSIIS", ...}` mapping protein names to protein sequences. This is what the output of the `simulate_msas` function looks like: there will be one sequence for each node in the input tree.

### Contact maps

A contact map file looks like this (taken from `tests/counting_tests/test_input_data/tiny/contact_map_dir/fam1.txt`):

```
5 sites
10101
01110
11110
01111
10011
```

The first line indicates the number of sites, and the matrix following indicates at position (i, j) whether positions i and j are in contact. This matrix is by definition symmetric.

### Site rates

A site rate file looks like this (taken from `tests/counting_tests/test_input_data/tiny/site_rates_dir/fam1.txt`):

```
5 sites
1.0 1.0 1.0 1.0 1.0
```

The first line indicates the number of sites, followed by the rate at which each site evolves (this is only used if a site is _not_ in contact with any other site; sites that are in contact will by simplicty all co-evolve at the same rate of 1.0).

## Additional data files

### Probability distributions

A  probability distribution file looks like this (taken from `tests/simulation_tests/test_input_data/normal_model/pi_2.txt`):

```
state   prob
SS  0.12
ST  0.16
TS  0.24
TT  0.48
```

The first column contains the state name (a string), and the second column contains the probability of that state. These numbers add up to 1.

### Rate matrices

A rate matrix file looks like this (taken from `tests/simulation_tests/test_input_data/normal_model/Q_2.txt`):

```
    SS  ST  TS  TT
SS  -4   2  2   0
ST  1.5   -3  0   1.5
TS  1   0   -2  1
TT  0   0.5   0.5   -1
```

Entry (i, i) indicates the negative of the rate at which state i is left, and entry (i, j) with i \neq j indicates the rate at which state i transitions to state j. Each row adds up to 0. For a rate matrix Q, the probability of being in state j at time t given that we started at time 0 in state i is given by \exp(tQ)_{i, j} where \exp is the matrix exponential. Note that this is a toy rate matrix, and that real co-evolution rate matrices will be of size 400 x 400.

### Count matrices

A count matrix file looks like this (taken from `tests/counting_tests/test_input_data/tiny/count_co_matrices_dir_edges/result.txt`):

```
2 matrices
16 states
1.99
    II  IL  IS  IT  LI  LL  LS  LT  SI  SL  SS  ST  TI  TL  TS  TT
II  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
IL  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
IS  0   0   0   0   0   0   0   2   0   0   0   0   0   0   0   0
IT  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
LI  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
LL  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
LS  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
LT  0   0   0   0   0   0   0   2   0   0   0   0   0   0   0   0
SI  0   0   0   0   0   0   0   0   0   0   0   0   0   2   0   0
SL  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
SS  0   0   0   0   0   0   0   0   0   0   0   0.5 0   0   0.5 0
ST  0   0   0   0   0   0   0   0   0   0   0   0.5 0   0   0   0
TI  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
TL  0   0   0   0   0   0   0   0   0   0   0   0   0   2   0   0
TS  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0.5 0
TT  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
5.01
    II  IL  IS  IT  LI  LL  LS  LT  SI  SL  SS  ST  TI  TL  TS  TT
II  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
IL  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
IS  0   0   2   0   0   0   0   0   0   0   0   0   0   0   0   0
IT  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
LI  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
LL  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
LS  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
LT  0   0   0   2   0   0   0   0   0   0   0   0   0   0   0   0
SI  0   0   0   0   0   0   0   0   2   0   0   0   0   0   0   0
SL  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
SS  0   0   0   0   0   0   0   0   0   0   1   0   0   0   0   0
ST  0   0   0   0   0   0   0   0   0   0   0   0.5 0   0   0   0
TI  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
TL  0   0   0   0   0   0   0   0   0   0   0   0   2   0   0   0
TS  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0.5 0
TT  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
```

The first row indicates how many matrices there are, the second row how many states there are, and then each count matrix follows, preceded by the branch length it is associated to. Note that this is a small example: on real data the count matrices will have size 400 x 400, and we will want to compute 100 of them. Count matrices are computed by the `count_co_transitions` function.
