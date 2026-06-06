import numpy as np
import scipy as sp

from naive_bayes.algorithms.base import Algorithm
from naive_bayes.algorithms.popularity import calculate_popularity
from naive_bayes.utils import get_top_ranks, spread_rows, validate_interaction_matrix


def calculate_weights(
    X: sp.sparse.csc_array,
    smoothing_strength: float,
    smoothing_type: str,
):
    """
    Calculates result[j, i] = P(j=1 | Y=i)
    So will be used like X @ result to get (u, i)-matrix
    """

    EMPIRICAL_SMOOTHING = 1.0

    # === input validation ===
    # check matrix (must be csc and binary) and convert to integer
    # this way, co-occurrence can be counted efficiently
    X = validate_interaction_matrix(X)
    user_count, item_count = X.shape

    # check smoothing
    if smoothing_strength < 0:
        raise ValueError("Smoothing must be non-negative")
    if smoothing_type not in ["uniform", "empirical1", "empirical2"]:
        raise ValueError(f"Unsupported smoothing type: `{smoothing_type}`")
    if smoothing_type in ["empirical1", "empirical2"] and smoothing_strength == 0:
        raise ValueError(f"Smoothing strength must be greater than 0 for empirical smoothing.")

    # === calculation ===
    # calculate pseudo counts
    if smoothing_type == "uniform":
        alpha = smoothing_strength * np.ones(item_count) + 1
        alpha_agg = alpha.sum() * np.ones(item_count)
    elif smoothing_type == "empirical1":
        empirical_counts = X.sum(axis=0) + EMPIRICAL_SMOOTHING
        empirical_means = empirical_counts / empirical_counts.sum()
        alpha = smoothing_strength * empirical_means + 1
        alpha_agg = alpha.sum() * np.ones(item_count)
    elif smoothing_type == "empirical2":
        empirical_counts = X.sum(axis=0) + EMPIRICAL_SMOOTHING
        empirical_means = empirical_counts / empirical_counts.sum()
        alpha = smoothing_strength * np.ones(item_count) + 1
        alpha_agg = smoothing_strength / empirical_means + item_count

    # calculate (co-)occurrence counts
    co_occurrence_counts = (X.T @ X).toarray()
    occurrence_counts = np.maximum(co_occurrence_counts.sum(axis=0) - co_occurrence_counts.diagonal(), 1)

    # calculate weight
    numerator = co_occurrence_counts + alpha[:, np.newaxis] - 1
    denominator = occurrence_counts[np.newaxis, :] + alpha_agg[:, np.newaxis] - item_count
    result = numerator / denominator
    np.fill_diagonal(result, 0)
    return result


class MultinomialNB(Algorithm):
    def __init__(
        self,
        bias_event_model: str,
        bias_weight: float,
        bias_weight_modifier: str,
        bias_smoothing: float,
        weights_smoothing_strength: float,
        weights_smoothing_type: str,
        centering: bool,
        absolute_pruning: bool,
        max_k: int | None
    ):
        super().__init__()

        # === check bias parameters ===
        if bias_event_model not in ["bernoulli", "multinomial"]:
            raise ValueError(f"Unknown bias event model: received `{bias_event_model}`.")

        if bias_weight < 0:
            raise ValueError(f"Bias weight must be non-negative: received `{bias_weight}`.")

        if bias_weight_modifier not in ["constant", "history", "loghistory"]:
            raise ValueError(f"Unknown bias weight modifier: received `{bias_weight_modifier}`.")

        if bias_smoothing < 0:
            raise ValueError(f"Bias smoothing must be non-negative: received `{bias_smoothing}`.")

        self.bias_event_model = bias_event_model
        self.bias_weight = bias_weight
        self.bias_weight_modifier = bias_weight_modifier
        self.bias_smoothing = bias_smoothing

        # === check weights parameter ===
        if weights_smoothing_strength == 0:
            weights_smoothing_type = "uniform"
        elif weights_smoothing_strength < 0:
            raise ValueError(
                f"Weights smoothing strength must be non-negative: received `{weights_smoothing_strength}`.")

        if weights_smoothing_type not in ["uniform", "empirical1", "empirical2"]:
            raise ValueError(f"Unknown weights smoothing type: received `{weights_smoothing_type}`.")

        self.weights_smoothing_strength = weights_smoothing_strength
        self.weights_smoothing_type = weights_smoothing_type

        # === check model parameter ===
        if max_k is not None and max_k <= 0:
            raise ValueError(f"Model max K must be strictly positive: received `{max_k}`.")

        self.centering = centering
        self.max_k = max_k
        self.absolute_pruning = absolute_pruning

    def _calculate_bias_weight(self, X):
        if self.bias_weight_modifier == "history":
            history_length = X.sum(axis=1)
            return self.bias_weight * history_length
        elif self.bias_weight_modifier == "loghistory":
            history_length = X.sum(axis=1)
            return self.bias_weight * np.log(history_length + 1)
        else:
            return self.bias_weight * np.ones(X.shape[0])

    def fit(self, X: sp.sparse.csr_array):
        X = sp.sparse.csc_array(X > 0)
        user_count, item_count = X.shape

        # === calculate bias ===
        popularity = calculate_popularity(X, self.bias_event_model, self.bias_smoothing)
        self.bias_ = np.log(popularity)

        # === calculate weights ===
        raw_weights = calculate_weights(
            X,
            self.weights_smoothing_strength,
            self.weights_smoothing_type,
        )
        self.weights_ = np.log(raw_weights, where=raw_weights > 0, out=np.zeros_like(raw_weights) - 1000)
        if self.centering:
            self.weights_ -= self.bias_[:, np.newaxis]

        # === prune weights ===
        if self.max_k is not None:
            if self.absolute_pruning:
                self.weights_ *= get_top_ranks(np.abs(self.weights_), self.max_k) > 0
            else:
                self.weights_ *= get_top_ranks(self.weights_, self.max_k) > 0

    def predict(self, X: sp.sparse.csr_array):
        # determine users for whom to calculate scores
        history_length = X.sum(axis=1)
        users = np.where(history_length > 0)[0]

        # calculate scores
        scores = (X[users] @ self.weights_) \
            + self._calculate_bias_weight(X)[users, np.newaxis] * self.bias_[np.newaxis, :]

        # reshape scores
        return spread_rows(scores, users, X.shape)
