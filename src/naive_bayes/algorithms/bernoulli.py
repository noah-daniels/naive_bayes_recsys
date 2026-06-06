import scipy as sp
import numpy as np
from naive_bayes.algorithms.base import Algorithm
from naive_bayes.algorithms.popularity import calculate_popularity
from naive_bayes.utils import get_top_ranks, spread_rows, validate_interaction_matrix


def calculate_weights(X: sp.sparse.csc_array, smoothing_strength: float, smoothing_type: str, complement_weight: float):
    """
    Calculates result1[j,i] = P(j=1|i=1) and result2[j,i] = P(j=1|i=0)

    + empirical1: fix trials: alpha + beta = s
    + empirical2: fix successes: alpha = s
    """

    EMPIRICAL_SMOOTHING = 1.0

    # === input validation ===
    # check matrix (must be csc and binary) and convert to integer
    # this way, co-occurrence can be counted efficiently
    X = validate_interaction_matrix(X)
    user_count, item_count = X.shape

    # check smoothing
    if smoothing_strength < 0:
        raise ValueError("Smoothing must be non-negative.")
    if smoothing_type not in ["uniform", "empirical1", "empirical2"]:
        raise ValueError(f"Unsupported smoothing type: `{smoothing_type}`.")
    if smoothing_type in ["empirical1", "empirical2"] and smoothing_strength == 0:
        raise ValueError(f"Smoothing strength must be greater than 0 for empirical smoothing.")

    # check complement weight
    if complement_weight < 0 or complement_weight > 1:
        raise ValueError("Complement weight must between 0 and 1.")

    # === calculations ===
    # calculate (co-)occurrence counts
    co_occurrence_counts = X.T @ X
    occurrence_counts = np.maximum(X.sum(axis=0), 1)

    # calculate pseudo counts
    if smoothing_type == "uniform":
        # +1 so smoothing strength = 0 corresponds to Beta(1,1) prior in MAP (which is constant)
        alpha = smoothing_strength * np.ones(item_count) + 1
        beta = smoothing_strength * np.ones(item_count) + 1
    elif smoothing_type == "empirical1":
        empirical_means = (occurrence_counts + EMPIRICAL_SMOOTHING) / (user_count + 2 * EMPIRICAL_SMOOTHING)
        alpha = smoothing_strength * empirical_means + 1
        beta = smoothing_strength * (1 - empirical_means) + 1
    elif smoothing_type == "empirical2":
        empirical_means = (occurrence_counts + EMPIRICAL_SMOOTHING) / (user_count + 2 * EMPIRICAL_SMOOTHING)
        alpha = smoothing_strength * np.ones(item_count) + 1
        beta = smoothing_strength * (1 / empirical_means - 1) + 1

    # calculate P(j=1 | i=1)
    numerator1 = co_occurrence_counts.toarray().astype(float) + (alpha[:, np.newaxis] - 1)
    denominator1 = (1 - complement_weight) * co_occurrence_counts.toarray() + complement_weight * \
        occurrence_counts[np.newaxis, :] + (alpha[:, np.newaxis] - 1) + (beta[:, np.newaxis] - 1)
    result1 = numerator1 / denominator1
    np.fill_diagonal(result1, 0)

    # # calculate P(j=1 | i=0)
    # numerator2 = occurrence_counts[:, np.newaxis] - co_occurrence_counts.toarray() + (alpha[:, np.newaxis] - 1)
    # denominator2 = (1 - complement_weight) * (occurrence_counts[:, np.newaxis] - co_occurrence_counts.toarray()) + complement_weight * (
    #     user_count - occurrence_counts[np.newaxis, :]) + (alpha[:, np.newaxis] - 1) + (beta[:, np.newaxis] - 1)
    # result2 = numerator2 / denominator2
    # np.fill_diagonal(result2, 0)

    return result1


class BernoulliNB(Algorithm):
    def __init__(
        self,
        bias_event_model: str,
        bias_weight: float,
        bias_weight_modifier: str,
        bias_smoothing: float,
        weights_smoothing_strength: float,
        weights_smoothing_type: str,
        weights_complement_weight: float,
        model_complement_weight: float,
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

        if weights_complement_weight < 0 or weights_complement_weight > 1:
            raise ValueError(
                f"Weights complement weight must be between 0 and 1: received `{weights_complement_weight}`.")

        self.weights_smoothing_strength = weights_smoothing_strength
        self.weights_smoothing_type = weights_smoothing_type
        self.weights_complement_weight = weights_complement_weight

        # === check model parameter ===
        if model_complement_weight < 0 or model_complement_weight > 1:
            raise ValueError(f"Model complement weight must be between 0 and 1: received `{model_complement_weight}`.")

        if max_k is not None and max_k <= 0:
            raise ValueError(f"Model max K must be strictly positive: received `{max_k}`.")

        self.model_complement_weight = model_complement_weight
        self.max_k = max_k

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
            self.weights_complement_weight,
        )
        log_weights = np.log(raw_weights, where=raw_weights > 0, out=np.zeros_like(raw_weights) - 1000)
        log_1mweights = np.log(1-raw_weights, where=(1-raw_weights) > 0, out=np.zeros_like(raw_weights) - 1000)

        # smoothing should ensure all entries are strictly positive, but the diagonal is always zero
        self.weights_ = log_weights - self.model_complement_weight * log_1mweights
        self.weights_adjustment_ = self.model_complement_weight * np.sum(log_1mweights, axis=0)

        # === prune weights ===
        if self.max_k is not None:
            self.weights_ *= get_top_ranks(self.weights_, self.max_k) > 0

    def predict(self, X: sp.sparse.csr_array):
        # determine users for whom to calculate scores
        history_length = X.sum(axis=1)
        users = np.where(history_length > 0)[0]

        # calculate scores
        scores = X[users] @ self.weights_ + self.weights_adjustment_[np.newaxis, :] + \
            self._calculate_bias_weight(X)[users, np.newaxis] * self.bias_[np.newaxis, :]

        # reshape scores
        return spread_rows(scores, users, X.shape)
