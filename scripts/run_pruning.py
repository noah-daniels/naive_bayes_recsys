import pathlib

import numpy as np

from naive_bayes.algorithms.bernoulli import BernoulliNB
from naive_bayes.algorithms.multinomial import MultinomialNB
from naive_bayes.datasets.amazon import load_amazon_books
from naive_bayes.datasets.mind import load_mind
from naive_bayes.datasets.movielens import load_ml25m
from naive_bayes.datasets.utils import create_interaction_array
from naive_bayes.pipeline import run_algorithm, strong_generalization
from naive_bayes.results import filter_df, get_best_params, load_results

PRIMARY_METRIC = "Recall@25"
RESULT_PATH = pathlib.Path("results")

DATASETS = {
    "Amazon Books": (load_amazon_books, "amazon_books"),
    "MIND": (lambda: load_mind()[0], "mind"),
    "ML25M": (lambda: load_ml25m()[0], "ml25m")
}

if __name__ == "__main__":
    df_ease, df_knn, df_bernoulli, df_multinomial = load_results(RESULT_PATH, include_pruning_results=False)

    for dataset_name, (load_fn, dataset_file_name) in DATASETS.items():
        X = create_interaction_array(load_fn())
        scenario = strong_generalization(X, seed_interactions=42, seed_users=42)

        params_bernoulli = get_best_params(filter_df(df_bernoulli, dataset=dataset_name), PRIMARY_METRIC)
        params_multinomial = get_best_params(filter_df(df_multinomial, dataset=dataset_name), PRIMARY_METRIC)

        params_bernoulli = [{**params_bernoulli, "max_k": i} for i in np.logspace(0.5, 3, 5).astype(int)]
        params_multinomial = [
            *[{**params_multinomial, "max_k": i, "centering": True, "absolute_pruning": True}
              for i in np.logspace(0.5, 3, 5).astype(int)],
            *[{**params_multinomial, "max_k": i, "centering": False, "absolute_pruning": True}
              for i in np.logspace(0.5, 3, 5).astype(int)],
            *[{**params_multinomial, "max_k": i, "centering": False,
               "absolute_pruning": False} for i in np.logspace(0.5, 3, 5).astype(int)]]

        run_algorithm(BernoulliNB, params_bernoulli, scenario, True).to_csv(
            RESULT_PATH / f"{dataset_file_name}_bernoulli_pruning.csv")
        run_algorithm(MultinomialNB, params_multinomial, scenario, True).to_csv(
            RESULT_PATH / f"{dataset_file_name}_multinomial_pruning1.csv")
