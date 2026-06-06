import scipy as sp
import numpy as np
from naive_bayes.algorithms.base import Algorithm
from naive_bayes.utils import get_top_ranks, spread_rows, validate_interaction_matrix


def calculate_popularity(X: sp.sparse.csc_array, event_model: str, smoothing: float):
    # === input validation ===
    # check matrix (must be csc and binary) and convert to integer
    X = validate_interaction_matrix(X)
    user_count, item_count = X.shape

    # check event model
    if event_model not in ["bernoulli", "multinomial"]:
        raise ValueError(f"Unsupported event model: `{event_model}`.")

    # check smoothing
    if smoothing < 0:
        raise ValueError("Smoothing must be non-negative.")

    # === calculations ===
    # calculate number of "successes" = weighted occurrence counts + smoothing pseudo-count
    successes = X.sum(axis=0) + smoothing

    # determine number of "trials" based on event model
    if event_model == "bernoulli":
        # smoothing multiplied by two because there are two classes: click and non-click
        trials = user_count + 2 * smoothing
    elif event_model == "multinomial":
        # smoothing is included in successes
        trials = successes.sum()

    # return result
    return successes / trials


class Popularity(Algorithm):
    def __init__(self, event_model: str, smoothing: float, max_k: int):
        super().__init__()

        if event_model not in ["bernoulli", "multinomial"]:
            raise ValueError(f"Unknown event model: received `{event_model}`.")

        if smoothing < 0:
            raise ValueError(f"Smoothing must be non-negative: received `{smoothing}`.")

        if max_k is not None and max_k <= 0:
            raise ValueError(f"Max k must be non-ngeative or None: received `{max_k}`.")

        self.event_model = event_model
        self.smoothing = smoothing
        self.max_k = max_k

    def fit(self, X: sp.sparse.csr_array):
        self.bias_ = calculate_popularity(sp.sparse.csc_array(X > 0), self.event_model, self.smoothing)
        self.bias_ *= get_top_ranks(self.bias_, self.max_k, axis=0) > 0

    def predict(self, X: sp.sparse.csr_array):
        users = np.where(X.sum(axis=1) > 0)[0]
        scores = np.tile(self.bias_, reps=(len(users), 1))
        return spread_rows(scores, users, X.shape)
