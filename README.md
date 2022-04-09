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
