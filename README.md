# cs267-project
CS267 Project: Scalable Estimation and Evaluation of Models of Amino-Acid Co-Evolution

To get set up on Cori:
```
module load python/3.8-anaconda-2020.11
```

Then install requirements:
```
pip install -r requirements.txt
```

To run ALL the tests:
```
python -m pytest tests/
```

To run JUST the counting tests (takes ~ 1 second):
```
python -m pytest tests/counting_tests/
```

To run JUST the simulations tests (takes ~ 10 seconds):
```
python -m pytest tests/simulation_tests/
```
