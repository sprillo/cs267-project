# CS267 Project: Scalable Estimation and Evaluation of Models of Amino-Acid Co-Evolution

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

## Main Modules

### src.counting

This module contains the functions to compute count matrices used to estimate rate matrices. The two functions exposed by this module are `count_transitions` and `count_co_transitions`. The function `count_transitions` counts transitions between _single_ amino acids, while the function `count_co_transitions` counts transitions between _pairs_ of amino acids. Our goal is to make `count_co_transitions` as fast as possible, but it might be easier to start with `count_transitions`.

### src.simulation

This module exposes a unique function `simulate_msas` which simulates data under a given Markov Chain model of amino acid evolution. Our goal is to make `simulate_msas` as fast as possible.

### src.evaluation

This module exposes a unique function `compute_log_likelihoods` which computes data log likelihood under a given Markov Chain model of amino acid evolution. This is the counterpart to `simulate_msas`, in the sense that `simulate_msas` can simulate data from the model, while `compute_log_likelihoods` can compute the likelihood of data under the model. Our goal is to make `compute_log_likelihoods` as fast as possible.

## Datasets

There are 4 kinds of data files:

- Trees
- Multiple sequence alignments (MSAs)
- Contact maps
- Site rates

Each protein family has exactly one file of each kind associated with it. We store trees, MSAs, contact maps, and site rates in separate directories, such as the tiny test dataset at "tests/counting_tests/test_input_data/tiny", which contains (among other test data):

- "tests/counting_tests/test_input_data/tiny/tree_dir"
- "tests/counting_tests/test_input_data/tiny/msa_dir"
- "tests/counting_tests/test_input_data/tiny/contact_map_dir"
- "tests/counting_tests/test_input_data/tiny/site_rates_dir"

Each protein family contains a file in each of these directories.

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

An MSA looks like this (taken from `tests/counting_tests/test_input_data/tiny/tree_dir/fam1.txt`):

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

This encodes a dictionary `{"internal-0": "SSIIS", "internal-1": "SSIIS", ...}` mapping protein names to protein sequences.

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

The first line indicates the number of sites, and the matrix following indicates at position $(i, j)$ whether positions $i$ and $j$ are in contact. This matrix is by definition symmetric.

### Site rates

A site rate file looks like this (taken from `tests/counting_tests/test_input_data/tiny/site_rates_dir/fam1.txt`):

```
5 sites
1.0 1.0 1.0 1.0 1.0
```

The first line indicates the number of sites, followed by the rate at which each site evolves (this is only used if a site is not in contact with any other site)

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

Entry $(i, i)$ indicates the negative of the rate at which state $i$ is left, and entry $(i, j)$ with $i \neq j$ indicates the rate at which state $i$ transitions to state $j$. Each row adds up to $0$. For a rate matrix $Q$, the probability of being in state $j$ at time $t$ given that we started at time $0$ in state $i$ is given by $\exp(tQ)_{i, j}$ where $\exp$ is the matrix exponential.
