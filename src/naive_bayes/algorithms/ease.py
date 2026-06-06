import numpy as np
import scipy as sp

from naive_bayes.algorithms.base import Algorithm
from naive_bayes.utils import spread_rows


class EASE(Algorithm):
    def __init__(self, l2):
        super().__init__()

        self.l2 = l2

    def fit(self, X: sp.sparse.csr_array):
        user_count, item_count = X.shape
        G = X.T @ X + self.l2 * np.eye(item_count)
        P = np.linalg.inv(G)
        B = - P @ np.diag(1.0 / np.diag(P))
        np.fill_diagonal(B, 0.0)

        self.B_ = B

    def predict(self, X: sp.sparse.csr_array):
        users = np.where(X.sum(axis=1) > 0)[0]
        scores = X[users] @ self.B_
        return spread_rows(scores, users, X.shape)
