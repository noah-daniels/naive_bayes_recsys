import numpy as np
import scipy as sp

import pytest

from naive_bayes.algorithms.popularity import calculate_popularity


@pytest.fixture
def interaction_matrix():
    return sp.sparse.csc_array(
        [
            [1, 0, 1, 0],
            [0, 1, 1, 0],
            [0, 1, 0, 1],
            [1, 0, 1, 1],
            [0, 0, 0, 1],
            [0, 1, 1, 0],
            [1, 0, 1, 1],
        ],
        dtype=bool,
    )


def test_popularity_bernoulli(interaction_matrix):
    for s in [0, 0.5, 1, 10]:
        pop = calculate_popularity(interaction_matrix, smoothing=s, event_model="bernoulli")

        assert pop.shape == (4,)
        assert pytest.approx(pop[0]) == (3 + s) / (7 + 2 * s)
        assert pytest.approx(pop[1]) == (3 + s) / (7 + 2 * s)
        assert pytest.approx(pop[2]) == (5 + s) / (7 + 2 * s)
        assert pytest.approx(pop[3]) == (4 + s) / (7 + 2 * s)


def test_popularity_multinomial(interaction_matrix):
    for s in [0, 0.5, 1, 10]:
        pop = calculate_popularity(interaction_matrix, smoothing=s, event_model="multinomial")

        assert pop.shape == (4,)
        assert pytest.approx(pop[0]) == (3 + s) / (15 + 4 * s)
        assert pytest.approx(pop[1]) == (3 + s) / (15 + 4 * s)
        assert pytest.approx(pop[2]) == (5 + s) / (15 + 4 * s)
        assert pytest.approx(pop[3]) == (4 + s) / (15 + 4 * s)
