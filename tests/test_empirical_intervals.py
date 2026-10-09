"""Statistical edge cases: trajectory SD, pointwise error and degenerate samples."""

import numpy as np
import pytest
from scipy.stats import binomtest, t

from scripts.empirical_intervals import (
    bootstrap_variance_interval,
    bounded_mean_interval,
    hoeffding_radius,
    student_mean_interval,
    wilson_interval,
)


def test_student_interval_uses_independent_runs_not_steps():
    data = np.array([[1, 2], [2, 4], [3, 6], [4, 8]])
    mean, low, high = student_mean_interval(data)
    expected = t.ppf(0.975, 3) * np.sqrt(5 / 3) / 2
    assert mean == pytest.approx([2.5, 5])
    assert high - mean == pytest.approx([expected, 2 * expected])
    assert mean - low == pytest.approx(high - mean)


def test_whole_trajectory_bootstrap_preserves_scaling_and_offset():
    data = np.array([[1, 12], [3, 16], [6, 22], [8, 26]], dtype=float)
    variance, low, high = bootstrap_variance_interval(data, rng=np.random.default_rng(42), resamples=1000)
    assert variance[1] == pytest.approx(4 * variance[0])
    assert low[1] == pytest.approx(4 * low[0])
    assert high[1] == pytest.approx(4 * high[0])
    shifted = bootstrap_variance_interval(data + 1e12, rng=np.random.default_rng(42), resamples=1000)
    for original, actual in zip((variance, low, high), shifted, strict=True):
        assert original == pytest.approx(actual)


def test_all_ones_keep_nonzero_distribution_free_interval():
    low, high = bounded_mean_interval(1.0, 100)
    assert low == pytest.approx(0.864189848425938)
    assert high == 1
    assert hoeffding_radius(200) == pytest.approx(0.09603227913199208)


@pytest.mark.parametrize("successes,total", [(0, 500), (1, 500), (235, 500), (500, 500)])
def test_wilson_against_scipy_binomial_interval(successes, total):
    expected = binomtest(successes, total).proportion_ci(method="wilson")
    assert wilson_interval(successes, total) == pytest.approx((expected.low, expected.high))


@pytest.mark.parametrize("values", [[], [[1]], [[1, float("nan")], [1, 2]], [1, 2]])
def test_invalid_independence_or_nonfinite_matrix_rejected(values):
    with pytest.raises(ValueError):
        student_mean_interval(values)


@pytest.mark.parametrize("successes,total", [(1, 0), (-1, 10), (11, 10), (0.5, 10)])
def test_impossible_binomial_counts_rejected(successes, total):
    with pytest.raises(ValueError):
        wilson_interval(successes, total)


def test_out_of_range_bounded_mean_rejected():
    with pytest.raises(ValueError):
        bounded_mean_interval(1.01, 100)


def test_bootstrap_requires_multiple_resamples():
    with pytest.raises(ValueError):
        bootstrap_variance_interval([[1], [2]], rng=np.random.default_rng(1), resamples=1)
