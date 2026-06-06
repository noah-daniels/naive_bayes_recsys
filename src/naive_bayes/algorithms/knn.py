import numpy as np
import scipy as sp
from sklearn.metrics.pairwise import cosine_similarity
from naive_bayes.algorithms.base import Algorithm
from naive_bayes.algorithms.popularity import calculate_popularity
from naive_bayes.utils import get_top_ranks, spread_rows


class ItemKNN(Algorithm):
    def __init__(self,
                 bias_event_model: str,
                 bias_weight: float,
                 bias_weight_modifier: str,
                 bias_smoothing: float,
                 similarity: str,
                 max_k: int
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

        # === check model parameter ===
        if similarity not in ["cosine", "conditional_probability"]:
            raise ValueError(f"Unknown similarity measure: received `{similarity}`.")

        if max_k is not None and max_k <= 0:
            raise ValueError(f"Max k must be strictly positive or None: received `{max_k}`.")

        self.similarity = similarity
        self.max_k = max_k

    def fit(self, X: sp.sparse.csr_array):
        X = sp.sparse.csc_array(X > 0)

        # calculate bias
        self.bias_ = np.log(calculate_popularity(X, self.bias_event_model, self.bias_smoothing))

        # calculate weights
        X = X.astype(np.int32)
        if self.similarity == "cosine":
            S = cosine_similarity(X.T, dense_output=True)
        elif self.similarity == "conditional_probability":
            S = (X.T @ X).toarray() / X.sum(axis=0)[:, np.newaxis]
        np.fill_diagonal(S, 0)

        # prune weights
        if self.max_k is not None:
            S *= get_top_ranks(S, self.max_k) > 0

        # set weights
        self.weights_ = S

    def predict(self, X: sp.sparse.csr_array):
        users = np.where(X.sum(axis=1) > 0)[0]
        scores = X[users] @ self.weights_ + \
            self._calculate_bias_weight(X)[users, np.newaxis] * self.bias_[np.newaxis, :]
        return spread_rows(scores, users, X.shape)

    def _calculate_bias_weight(self, X):
        if self.bias_weight_modifier == "history":
            history_length = X.sum(axis=1)
            return self.bias_weight * history_length
        elif self.bias_weight_modifier == "loghistory":
            history_length = X.sum(axis=1)
            return self.bias_weight * np.log(history_length + 1)
        else:
            return self.bias_weight * np.ones(X.shape[0])
