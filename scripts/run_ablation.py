import pathlib
import numpy as np
import scipy as sp
from naive_bayes.algorithms.ease import EASE
from naive_bayes.datasets import load_amazon_books, load_ml25m, load_mind
from naive_bayes.datasets.utils import create_interaction_array
from naive_bayes.pipeline import strong_generalization, run_algorithm
from naive_bayes.algorithms.bernoulli import BernoulliNB
from naive_bayes.algorithms.multinomial import MultinomialNB
from naive_bayes.algorithms.knn import ItemKNN


RESULT_PATH = pathlib.Path("results")


def get_knn_params():
    params = []
    for bias_event_model in ["bernoulli", "multinomial"]:
        for bias_weight_modifier in ["constant", "history"]:
            for similarity in ["cosine", "conditional_probability"]:
                for bias_weight in [0, *np.logspace(-2, 2, 10)]:
                    for max_k in [None, *np.logspace(0.5, 3, 5).astype(int)]:
                        params.append({
                            "bias_event_model": bias_event_model,
                            "bias_weight": bias_weight,
                            "bias_weight_modifier": bias_weight_modifier,
                            "bias_smoothing": 1.0,
                            "similarity": similarity,
                            "max_k": max_k
                        })
    return params


def get_bernoulli_params():
    params = []
    for bias_event_model in ["bernoulli"]:
        for bias_weight_modifier in ["constant", "history"]:
            for weights_smoothing_type in ["uniform", "empirical1", "empirical2", None]:
                for bias_weight in [0, *np.logspace(-2, 2, 10)]:
                    for weights_smoothing_strength in np.logspace(-2, 4, 10) if weights_smoothing_type is not None else [0]:
                        for complement_weight in [None, *sp.special.expit(np.linspace(-4, 4, 9))]:
                            if bias_weight == 0 and bias_weight_modifier == "history":
                                continue

                            params.append({
                                "bias_event_model": bias_event_model,
                                "bias_weight": bias_weight,
                                "bias_weight_modifier": bias_weight_modifier,
                                "bias_smoothing": 1.0,
                                "weights_smoothing_strength": weights_smoothing_strength,
                                "weights_smoothing_type": weights_smoothing_type if weights_smoothing_type is not None else "uniform",
                                "weights_complement_weight": complement_weight if complement_weight is not None else 1,
                                "model_complement_weight": complement_weight if complement_weight is not None else 0,
                                "max_k": None
                            })
    return params


def get_multinomial_params():
    params = []
    for bias_event_model in ["bernoulli", "multinomial"]:
        for bias_weight_modifier in ["constant", "history"]:
            for weights_smoothing_type in ["uniform", "empirical1", "empirical2", None]:
                for bias_weight in [0, *np.logspace(-2, 2, 10)]:
                    for weights_smoothing_strength in np.logspace(-2, 4, 10) if weights_smoothing_type is not None else [0]:
                        if bias_weight == 0 and bias_weight_modifier == "history":
                            continue

                        params.append({
                            "bias_event_model": bias_event_model,
                            "bias_weight": bias_weight,
                            "bias_weight_modifier": bias_weight_modifier,
                            "bias_smoothing": 1.0,
                            "weights_smoothing_strength": weights_smoothing_strength,
                            "weights_smoothing_type": weights_smoothing_type if weights_smoothing_type is not None else "uniform",
                            "max_k": None,
                            "absolute_pruning": True,
                            "centering": False,
                        })
    return params


def get_ease_params():
    params = []
    for l2 in np.logspace(-3, 3, 10):
        params.append({
            "l2": l2
        })
    return params


def run(scenario, dataset_name):
    run_algorithm(ItemKNN, get_knn_params(), scenario, True).to_csv(
        RESULT_PATH / f"{dataset_name}_knn.csv")
    run_algorithm(MultinomialNB, get_multinomial_params(), scenario, True).to_csv(
        RESULT_PATH / f"{dataset_name}_multinomial.csv")
    run_algorithm(BernoulliNB, get_bernoulli_params(), scenario, True).to_csv(
        RESULT_PATH / f"{dataset_name}_bernoulli.csv")
    run_algorithm(EASE, get_ease_params(), scenario, True).to_csv(
        RESULT_PATH / f"{dataset_name}_ease.csv")


if __name__ == "__main__":
    # print("EASE" len(get_ease_params()))
    # print("kNN" len(get_knn_params()))
    # print("Bernoulli" len(get_bernoulli_params()))
    # print("Multinomial" len(get_multinomial_params()))

    X = create_interaction_array(load_amazon_books())
    scenario = strong_generalization(X, seed_interactions=42, seed_users=42)
    run(scenario, "amazon_books")

    X = create_interaction_array(load_mind()[0])
    scenario = strong_generalization(X, seed_interactions=42, seed_users=42)
    run(scenario, "mind")

    X = create_interaction_array(load_ml25m()[0])
    scenario = strong_generalization(X, seed_interactions=42, seed_users=42)
    run(scenario, "ml25m")
