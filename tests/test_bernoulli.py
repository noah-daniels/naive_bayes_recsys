import numpy as np
import scipy as sp

import pytest

from naive_bayes.algorithms.bernoulli import calculate_weights


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
    for w in [0.1, 0.5, 0.9, 1]:
        weights1, weights2 = calculate_weights(
            interaction_matrix,
            smoothing_strength=0,
            smoothing_type="uniform",
            complement_weight=w,
        )

        assert weights1.shape == (4, 4)
        assert weights2.shape == (4, 4)
        assert np.all(np.diagonal(weights1) == 0)
        assert np.all(np.diagonal(weights2) == 0)

        assert pytest.approx(weights1[0, 1]) == 0 / ((1 - w) * 0 + w * 3)
        assert pytest.approx(weights1[0, 2]) == 3 / ((1 - w) * 3 + w * 5)
        assert pytest.approx(weights1[0, 3]) == 2 / ((1 - w) * 2 + w * 4)
        assert pytest.approx(weights1[1, 0]) == 0 / ((1 - w) * 0 + w * 3)
        assert pytest.approx(weights1[1, 2]) == 2 / ((1 - w) * 2 + w * 5)
        assert pytest.approx(weights1[1, 3]) == 1 / ((1 - w) * 1 + w * 4)
        assert pytest.approx(weights1[2, 0]) == 3 / ((1 - w) * 3 + w * 3)
        assert pytest.approx(weights1[2, 1]) == 2 / ((1 - w) * 2 + w * 3)
        assert pytest.approx(weights1[2, 3]) == 2 / ((1 - w) * 2 + w * 4)
        assert pytest.approx(weights1[3, 0]) == 2 / ((1 - w) * 2 + w * 3)
        assert pytest.approx(weights1[3, 1]) == 1 / ((1 - w) * 1 + w * 3)
        assert pytest.approx(weights1[3, 2]) == 2 / ((1 - w) * 2 + w * 5)

        assert pytest.approx(weights2[0, 1]) == 3 / ((1 - w) * 3 + w * 4)
        assert pytest.approx(weights2[0, 2]) == 0 / ((1 - w) * 0 + w * 2)
        assert pytest.approx(weights2[0, 3]) == 1 / ((1 - w) * 1 + w * 3)
        assert pytest.approx(weights2[1, 0]) == 3 / ((1 - w) * 3 + w * 4)
        assert pytest.approx(weights2[1, 2]) == 1 / ((1 - w) * 1 + w * 2)
        assert pytest.approx(weights2[1, 3]) == 2 / ((1 - w) * 2 + w * 3)
        assert pytest.approx(weights2[2, 0]) == 2 / ((1 - w) * 2 + w * 4)
        assert pytest.approx(weights2[2, 1]) == 3 / ((1 - w) * 3 + w * 4)
        assert pytest.approx(weights2[2, 3]) == 3 / ((1 - w) * 3 + w * 3)
        assert pytest.approx(weights2[3, 0]) == 2 / ((1 - w) * 2 + w * 4)
        assert pytest.approx(weights2[3, 1]) == 3 / ((1 - w) * 3 + w * 4)
        assert pytest.approx(weights2[3, 2]) == 2 / ((1 - w) * 2 + w * 2)


def test_weights_uniform(interaction_matrix):
    for w in [0, 0.1, 0.5, 0.9, 1]:
        for s in [1, 2, 5, 10]:
            weights1, weights2 = calculate_weights(
                interaction_matrix,
                smoothing_strength=s,
                smoothing_type="uniform",
                complement_weight=w,
            )

            assert weights1.shape == (4, 4)
            assert weights2.shape == (4, 4)
            assert np.all(np.diagonal(weights1) == 0)
            assert np.all(np.diagonal(weights2) == 0)

            assert pytest.approx(weights1[0, 1]) == (0 + s) / ((1 - w) * 0 + w * 3 + 2 * s)
            assert pytest.approx(weights1[0, 2]) == (3 + s) / ((1 - w) * 3 + w * 5 + 2 * s)
            assert pytest.approx(weights1[0, 3]) == (2 + s) / ((1 - w) * 2 + w * 4 + 2 * s)
            assert pytest.approx(weights1[1, 0]) == (0 + s) / ((1 - w) * 0 + w * 3 + 2 * s)
            assert pytest.approx(weights1[1, 2]) == (2 + s) / ((1 - w) * 2 + w * 5 + 2 * s)
            assert pytest.approx(weights1[1, 3]) == (1 + s) / ((1 - w) * 1 + w * 4 + 2 * s)
            assert pytest.approx(weights1[2, 0]) == (3 + s) / ((1 - w) * 3 + w * 3 + 2 * s)
            assert pytest.approx(weights1[2, 1]) == (2 + s) / ((1 - w) * 2 + w * 3 + 2 * s)
            assert pytest.approx(weights1[2, 3]) == (2 + s) / ((1 - w) * 2 + w * 4 + 2 * s)
            assert pytest.approx(weights1[3, 0]) == (2 + s) / ((1 - w) * 2 + w * 3 + 2 * s)
            assert pytest.approx(weights1[3, 1]) == (1 + s) / ((1 - w) * 1 + w * 3 + 2 * s)
            assert pytest.approx(weights1[3, 2]) == (2 + s) / ((1 - w) * 2 + w * 5 + 2 * s)

            assert pytest.approx(weights2[0, 1]) == (3 + s) / ((1 - w) * 3 + w * 4 + 2 * s)
            assert pytest.approx(weights2[0, 2]) == (0 + s) / ((1 - w) * 0 + w * 2 + 2 * s)
            assert pytest.approx(weights2[0, 3]) == (1 + s) / ((1 - w) * 1 + w * 3 + 2 * s)
            assert pytest.approx(weights2[1, 0]) == (3 + s) / ((1 - w) * 3 + w * 4 + 2 * s)
            assert pytest.approx(weights2[1, 2]) == (1 + s) / ((1 - w) * 1 + w * 2 + 2 * s)
            assert pytest.approx(weights2[1, 3]) == (2 + s) / ((1 - w) * 2 + w * 3 + 2 * s)
            assert pytest.approx(weights2[2, 0]) == (2 + s) / ((1 - w) * 2 + w * 4 + 2 * s)
            assert pytest.approx(weights2[2, 1]) == (3 + s) / ((1 - w) * 3 + w * 4 + 2 * s)
            assert pytest.approx(weights2[2, 3]) == (3 + s) / ((1 - w) * 3 + w * 3 + 2 * s)
            assert pytest.approx(weights2[3, 0]) == (2 + s) / ((1 - w) * 2 + w * 4 + 2 * s)
            assert pytest.approx(weights2[3, 1]) == (3 + s) / ((1 - w) * 3 + w * 4 + 2 * s)
            assert pytest.approx(weights2[3, 2]) == (2 + s) / ((1 - w) * 2 + w * 2 + 2 * s)


def test_weights_empirical1(interaction_matrix):
    for w in [0, 0.1, 0.5, 0.9, 1]:
        for s in [0.5, 1, 2, 5, 10]:
            weights1, weights2 = calculate_weights(
                interaction_matrix,
                smoothing_strength=s,
                smoothing_type="empirical1",
                complement_weight=w,
            )

            assert weights1.shape == (4, 4)
            assert weights2.shape == (4, 4)
            assert np.all(np.diagonal(weights1) == 0)
            assert np.all(np.diagonal(weights2) == 0)

            EMPIRICAL_SMOOTHING = 1.0
            p0 = (3 + EMPIRICAL_SMOOTHING) / (7 + 2 * EMPIRICAL_SMOOTHING)
            p1 = (3 + EMPIRICAL_SMOOTHING) / (7 + 2 * EMPIRICAL_SMOOTHING)
            p2 = (5 + EMPIRICAL_SMOOTHING) / (7 + 2 * EMPIRICAL_SMOOTHING)
            p3 = (4 + EMPIRICAL_SMOOTHING) / (7 + 2 * EMPIRICAL_SMOOTHING)

            assert pytest.approx(weights1[0, 1]) == (0 + s * p0) / ((1 - w) * 0 + w * 3 + s)
            assert pytest.approx(weights1[0, 2]) == (3 + s * p0) / ((1 - w) * 3 + w * 5 + s)
            assert pytest.approx(weights1[0, 3]) == (2 + s * p0) / ((1 - w) * 2 + w * 4 + s)
            assert pytest.approx(weights1[1, 0]) == (0 + s * p1) / ((1 - w) * 0 + w * 3 + s)
            assert pytest.approx(weights1[1, 2]) == (2 + s * p1) / ((1 - w) * 2 + w * 5 + s)
            assert pytest.approx(weights1[1, 3]) == (1 + s * p1) / ((1 - w) * 1 + w * 4 + s)
            assert pytest.approx(weights1[2, 0]) == (3 + s * p2) / ((1 - w) * 3 + w * 3 + s)
            assert pytest.approx(weights1[2, 1]) == (2 + s * p2) / ((1 - w) * 2 + w * 3 + s)
            assert pytest.approx(weights1[2, 3]) == (2 + s * p2) / ((1 - w) * 2 + w * 4 + s)
            assert pytest.approx(weights1[3, 0]) == (2 + s * p3) / ((1 - w) * 2 + w * 3 + s)
            assert pytest.approx(weights1[3, 1]) == (1 + s * p3) / ((1 - w) * 1 + w * 3 + s)
            assert pytest.approx(weights1[3, 2]) == (2 + s * p3) / ((1 - w) * 2 + w * 5 + s)

            assert pytest.approx(weights2[0, 1]) == (3 + s * p0) / ((1 - w) * 3 + w * 4 + s)
            assert pytest.approx(weights2[0, 2]) == (0 + s * p0) / ((1 - w) * 0 + w * 2 + s)
            assert pytest.approx(weights2[0, 3]) == (1 + s * p0) / ((1 - w) * 1 + w * 3 + s)
            assert pytest.approx(weights2[1, 0]) == (3 + s * p1) / ((1 - w) * 3 + w * 4 + s)
            assert pytest.approx(weights2[1, 2]) == (1 + s * p1) / ((1 - w) * 1 + w * 2 + s)
            assert pytest.approx(weights2[1, 3]) == (2 + s * p1) / ((1 - w) * 2 + w * 3 + s)
            assert pytest.approx(weights2[2, 0]) == (2 + s * p2) / ((1 - w) * 2 + w * 4 + s)
            assert pytest.approx(weights2[2, 1]) == (3 + s * p2) / ((1 - w) * 3 + w * 4 + s)
            assert pytest.approx(weights2[2, 3]) == (3 + s * p2) / ((1 - w) * 3 + w * 3 + s)
            assert pytest.approx(weights2[3, 0]) == (2 + s * p3) / ((1 - w) * 2 + w * 4 + s)
            assert pytest.approx(weights2[3, 1]) == (3 + s * p3) / ((1 - w) * 3 + w * 4 + s)
            assert pytest.approx(weights2[3, 2]) == (2 + s * p3) / ((1 - w) * 2 + w * 2 + s)


def test_weights_empirical2(interaction_matrix):
    for w in [0, 0.1, 0.5, 0.9, 1]:
        for s in [0.5, 1, 2, 5, 10]:
            weights1, weights2 = calculate_weights(
                interaction_matrix,
                smoothing_strength=s,
                smoothing_type="empirical2",
                complement_weight=w,
            )

            assert weights1.shape == (4, 4)
            assert weights2.shape == (4, 4)
            assert np.all(np.diagonal(weights1) == 0)
            assert np.all(np.diagonal(weights2) == 0)

            EMPIRICAL_SMOOTHING = 1.0
            p0 = (3 + EMPIRICAL_SMOOTHING) / (7 + 2 * EMPIRICAL_SMOOTHING)
            p1 = (3 + EMPIRICAL_SMOOTHING) / (7 + 2 * EMPIRICAL_SMOOTHING)
            p2 = (5 + EMPIRICAL_SMOOTHING) / (7 + 2 * EMPIRICAL_SMOOTHING)
            p3 = (4 + EMPIRICAL_SMOOTHING) / (7 + 2 * EMPIRICAL_SMOOTHING)

            assert pytest.approx(weights1[0, 1]) == (0 + s) / ((1 - w) * 0 + w * 3 + s / p0)
            assert pytest.approx(weights1[0, 2]) == (3 + s) / ((1 - w) * 3 + w * 5 + s / p0)
            assert pytest.approx(weights1[0, 3]) == (2 + s) / ((1 - w) * 2 + w * 4 + s / p0)
            assert pytest.approx(weights1[1, 0]) == (0 + s) / ((1 - w) * 0 + w * 3 + s / p1)
            assert pytest.approx(weights1[1, 2]) == (2 + s) / ((1 - w) * 2 + w * 5 + s / p1)
            assert pytest.approx(weights1[1, 3]) == (1 + s) / ((1 - w) * 1 + w * 4 + s / p1)
            assert pytest.approx(weights1[2, 0]) == (3 + s) / ((1 - w) * 3 + w * 3 + s / p2)
            assert pytest.approx(weights1[2, 1]) == (2 + s) / ((1 - w) * 2 + w * 3 + s / p2)
            assert pytest.approx(weights1[2, 3]) == (2 + s) / ((1 - w) * 2 + w * 4 + s / p2)
            assert pytest.approx(weights1[3, 0]) == (2 + s) / ((1 - w) * 2 + w * 3 + s / p3)
            assert pytest.approx(weights1[3, 1]) == (1 + s) / ((1 - w) * 1 + w * 3 + s / p3)
            assert pytest.approx(weights1[3, 2]) == (2 + s) / ((1 - w) * 2 + w * 5 + s / p3)

            assert pytest.approx(weights2[0, 1]) == (3 + s) / ((1 - w) * 3 + w * 4 + s / p0)
            assert pytest.approx(weights2[0, 2]) == (0 + s) / ((1 - w) * 0 + w * 2 + s / p0)
            assert pytest.approx(weights2[0, 3]) == (1 + s) / ((1 - w) * 1 + w * 3 + s / p0)
            assert pytest.approx(weights2[1, 0]) == (3 + s) / ((1 - w) * 3 + w * 4 + s / p1)
            assert pytest.approx(weights2[1, 2]) == (1 + s) / ((1 - w) * 1 + w * 2 + s / p1)
            assert pytest.approx(weights2[1, 3]) == (2 + s) / ((1 - w) * 2 + w * 3 + s / p1)
            assert pytest.approx(weights2[2, 0]) == (2 + s) / ((1 - w) * 2 + w * 4 + s / p2)
            assert pytest.approx(weights2[2, 1]) == (3 + s) / ((1 - w) * 3 + w * 4 + s / p2)
            assert pytest.approx(weights2[2, 3]) == (3 + s) / ((1 - w) * 3 + w * 3 + s / p2)
            assert pytest.approx(weights2[3, 0]) == (2 + s) / ((1 - w) * 2 + w * 4 + s / p3)
            assert pytest.approx(weights2[3, 1]) == (3 + s) / ((1 - w) * 3 + w * 4 + s / p3)
            assert pytest.approx(weights2[3, 2]) == (2 + s) / ((1 - w) * 2 + w * 2 + s / p3)
