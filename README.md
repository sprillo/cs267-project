# CS267 Project: Scalable Estimation and Evaluation of Models of Amino-Acid Co-Evolution

To get set up on Cori:
```
module load python/3.8-anaconda-2020.11
```

Then install requirements:
```
pip install -r requirements.txt
```

To run the counting fast tests (takes ~ 1 second):
```
python -m pytest tests/counting_tests/
```

To run the counting fast AND SLOW tests (takes ~ 5 minutes, requires 32 cores):
```
python -m pytest tests/counting_tests/ --runslow
```

To run the simulation fast tests (takes ~ 10 seconds):
```
python -m pytest tests/simulation_tests/
```
