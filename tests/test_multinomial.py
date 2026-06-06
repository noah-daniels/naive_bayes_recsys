import numpy as np
import scipy as sp

import pytest

from naive_bayes.algorithms.multinomial import calculate_weights


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


def test_weights_raw(interaction_matrix):
    weights = calculate_weights(interaction_matrix, smoothing_strength=0, smoothing_type="uniform")

    assert weights.shape == (4, 4)
    assert np.allclose(weights.sum(axis=0), 1)
    assert np.all(np.diagonal(weights) == 0)

    assert pytest.approx(weights[0, 1]) == 0 / 3
    assert pytest.approx(weights[0, 2]) == 3 / 7
    assert pytest.approx(weights[0, 3]) == 2 / 5
    assert pytest.approx(weights[1, 0]) == 0 / 5
    assert pytest.approx(weights[1, 2]) == 2 / 7
    assert pytest.approx(weights[1, 3]) == 1 / 5
    assert pytest.approx(weights[2, 0]) == 3 / 5
    assert pytest.approx(weights[2, 1]) == 2 / 3
    assert pytest.approx(weights[2, 3]) == 2 / 5
    assert pytest.approx(weights[3, 0]) == 2 / 5
    assert pytest.approx(weights[3, 1]) == 1 / 3
    assert pytest.approx(weights[3, 2]) == 2 / 7


def test_weights_uniform(interaction_matrix):
    for s in [1, 2, 10]:
        weights = calculate_weights(interaction_matrix, smoothing_strength=s, smoothing_type="uniform")

        assert weights.shape == (4, 4)
        assert np.all(np.diagonal(weights) == 0)

        assert pytest.approx(weights[0, 1]) == (0 + s) / (3 + 4 * s)
        assert pytest.approx(weights[0, 2]) == (3 + s) / (7 + 4 * s)
        assert pytest.approx(weights[0, 3]) == (2 + s) / (5 + 4 * s)
        assert pytest.approx(weights[1, 0]) == (0 + s) / (5 + 4 * s)
        assert pytest.approx(weights[1, 2]) == (2 + s) / (7 + 4 * s)
        assert pytest.approx(weights[1, 3]) == (1 + s) / (5 + 4 * s)
        assert pytest.approx(weights[2, 0]) == (3 + s) / (5 + 4 * s)
        assert pytest.approx(weights[2, 1]) == (2 + s) / (3 + 4 * s)
        assert pytest.approx(weights[2, 3]) == (2 + s) / (5 + 4 * s)
        assert pytest.approx(weights[3, 0]) == (2 + s) / (5 + 4 * s)
        assert pytest.approx(weights[3, 1]) == (1 + s) / (3 + 4 * s)
        assert pytest.approx(weights[3, 2]) == (2 + s) / (7 + 4 * s)


def test_weights_empirical1(interaction_matrix):
    for s in [0.5, 1, 2, 10]:
        weights = calculate_weights(interaction_matrix, smoothing_strength=s, smoothing_type="empirical1")

        assert weights.shape == (4, 4)
        assert np.all(np.diagonal(weights) == 0)

        EMPIRICAL_SMOOTHING = 1.0
        p0 = (3 + EMPIRICAL_SMOOTHING) / (15 + 4 * EMPIRICAL_SMOOTHING)
        p1 = (3 + EMPIRICAL_SMOOTHING) / (15 + 4 * EMPIRICAL_SMOOTHING)
        p2 = (5 + EMPIRICAL_SMOOTHING) / (15 + 4 * EMPIRICAL_SMOOTHING)
        p3 = (4 + EMPIRICAL_SMOOTHING) / (15 + 4 * EMPIRICAL_SMOOTHING)

        assert pytest.approx(weights[0, 1]) == (0 + s * p0) / (3 + s)
        assert pytest.approx(weights[0, 2]) == (3 + s * p0) / (7 + s)
        assert pytest.approx(weights[0, 3]) == (2 + s * p0) / (5 + s)
        assert pytest.approx(weights[1, 0]) == (0 + s * p1) / (5 + s)
        assert pytest.approx(weights[1, 2]) == (2 + s * p1) / (7 + s)
        assert pytest.approx(weights[1, 3]) == (1 + s * p1) / (5 + s)
        assert pytest.approx(weights[2, 0]) == (3 + s * p2) / (5 + s)
        assert pytest.approx(weights[2, 1]) == (2 + s * p2) / (3 + s)
        assert pytest.approx(weights[2, 3]) == (2 + s * p2) / (5 + s)
        assert pytest.approx(weights[3, 0]) == (2 + s * p3) / (5 + s)
        assert pytest.approx(weights[3, 1]) == (1 + s * p3) / (3 + s)
        assert pytest.approx(weights[3, 2]) == (2 + s * p3) / (7 + s)


def test_weights_empirical2(interaction_matrix):
    for s in [0.5, 1, 2, 10]:
        weights = calculate_weights(interaction_matrix, smoothing_strength=s, smoothing_type="empirical2")

        assert weights.shape == (4, 4)
        assert np.all(np.diagonal(weights) == 0)

        EMPIRICAL_SMOOTHING = 1.0
        p0 = (3 + EMPIRICAL_SMOOTHING) / (15 + 4 * EMPIRICAL_SMOOTHING)
        p1 = (3 + EMPIRICAL_SMOOTHING) / (15 + 4 * EMPIRICAL_SMOOTHING)
        p2 = (5 + EMPIRICAL_SMOOTHING) / (15 + 4 * EMPIRICAL_SMOOTHING)
        p3 = (4 + EMPIRICAL_SMOOTHING) / (15 + 4 * EMPIRICAL_SMOOTHING)

        assert pytest.approx(weights[0, 1]) == (0 + s) / (3 + s / p0)
        assert pytest.approx(weights[0, 2]) == (3 + s) / (7 + s / p0)
        assert pytest.approx(weights[0, 3]) == (2 + s) / (5 + s / p0)
        assert pytest.approx(weights[1, 0]) == (0 + s) / (5 + s / p1)
        assert pytest.approx(weights[1, 2]) == (2 + s) / (7 + s / p1)
        assert pytest.approx(weights[1, 3]) == (1 + s) / (5 + s / p1)
        assert pytest.approx(weights[2, 0]) == (3 + s) / (5 + s / p2)
        assert pytest.approx(weights[2, 1]) == (2 + s) / (3 + s / p2)
        assert pytest.approx(weights[2, 3]) == (2 + s) / (5 + s / p2)
        assert pytest.approx(weights[3, 0]) == (2 + s) / (5 + s / p3)
        assert pytest.approx(weights[3, 1]) == (1 + s) / (3 + s / p3)
        assert pytest.approx(weights[3, 2]) == (2 + s) / (7 + s / p3)
