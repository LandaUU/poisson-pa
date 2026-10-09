"""Pointwise conditional Monte Carlo intervals, without network-level inference."""

import numpy as np
from scipy.stats import norm, t


def trajectories(values):
    data = np.asarray(values, dtype=float)
    if data.ndim != 2 or data.shape[0] < 2 or data.shape[1] < 1 or not np.isfinite(data).all():
        raise ValueError("Expected a finite runs-by-steps matrix with at least two independent runs")
    return data


def student_mean_interval(values):
    """Approximate 95% Student interval; complete rows are independent runs."""
    data = trajectories(values)
    mean = data.mean(axis=0)
    half = t.ppf(0.975, data.shape[0] - 1) * data.std(axis=0, ddof=1) / np.sqrt(data.shape[0])
    return mean, mean - half, mean + half


def bootstrap_variance_interval(values, *, rng, resamples=2000):
    """Approximate percentile interval, resampling whole rows jointly across time."""
    data = trajectories(values)
    if resamples < 2:
        raise ValueError("At least two bootstrap resamples are required")
    runs = data.shape[0]
    weights = rng.multinomial(runs, np.full(runs, 1 / runs), size=resamples)
    centered = data - data.mean(axis=0)
    first = weights @ centered / runs
    second = weights @ (centered * centered) / runs
    bootstrap = np.maximum(0, (second - first * first) * runs / (runs - 1))
    low, high = np.quantile(bootstrap, (0.025, 0.975), axis=0)
    return data.var(axis=0, ddof=1), low, high


def hoeffding_radius(runs):
    if not isinstance(runs, (int, np.integer)) or runs < 1:
        raise ValueError("runs must be a positive integer")
    return float(np.sqrt(np.log(40) / (2 * runs)))


def bounded_mean_interval(mean, runs):
    values = np.asarray(mean, dtype=float)
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError("Means must be finite and in [0, 1]")
    radius = hoeffding_radius(runs)
    return np.maximum(0, values - radius), np.minimum(1, values + radius)


def wilson_interval(successes, replications):
    if not isinstance(successes, (int, np.integer)) or not isinstance(replications, (int, np.integer)):
        raise ValueError("Binomial counts must be integers")
    if replications < 1 or not 0 <= successes <= replications:
        raise ValueError("Require 0 <= successes <= replications and replications > 0")
    z = norm.ppf(0.975)
    q = successes / replications
    denominator = 1 + z * z / replications
    center = (q + z * z / (2 * replications)) / denominator
    half = z * np.sqrt(q * (1 - q) / replications + z * z / (4 * replications**2)) / denominator
    return max(0.0, center - half), min(1.0, center + half)
