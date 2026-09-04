# Theoretically Sound Item Similarity Measures for Recommender Systems

## Overview

- See `src/naive_bayes/algorithms` for implementations.
- See `notebooks/results.ipynb` for code used to generate plots.
- See `results/` for all results (main results and results of of the ablation study).
- See `scripts/` for code to run the ablation study and final evaluation.

## Instalation

1. Create viertual environment `python3 -m venv env`
1. Install project and dependencies using `pip3 install -e .`
1. Download datasets to `data/ml`, `data/amazon` and `data/mind`.
    - MovieLens can be found [here](https://grouplens.org/datasets/movielens/) (download 25M version)
    - MIND can be found [here](https://msnews.github.io/) (download train and validation sets)
    - Amazon Books can be found [here](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/) (download the 5-core version)
1. Copy `.env.example` and rename it to `.env`, change data path if desired.

## Run experiments

+ `python3 scripts/run_ablation.py` will run the ablation study an create csv files in the `results/` directory for each (dataset, algorithm) combination
+ `python3 scripts/run_pruning.py` will take the best parameters according the csv files in the results directory and try various pruning approaches. Results are also saved in the `results/` directory.
+ `python3 scripts/run_final.py` will take the best parameters according to the csv files in the results directory and run them 5 times on the test dataset. Resultsa are saved in the `results/` directory.

First start up will take longer as the dataset are parsed and processed. Subsequest runs will start nearly immediately as the processed datasets are saved.

Running the full ablations study can take a couple days. Running the final experiments should finish within the hour. 
