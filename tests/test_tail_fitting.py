from itertools import permutations

import numpy as np
import pytest

from model_fitting import _ks_tail_distance, hill_tail_fit


def independent_distance(ratios, index):
    points = sorted({1.0, *ratios})
    candidates = []
    for point in points:
        model = point ** (-index)
        candidates.append(abs(sum(x > point for x in ratios) / len(ratios) - model))
        if point > 1:
            candidates.append(abs(sum(x >= point for x in ratios) / len(ratios) - model))
    return max(candidates)


@pytest.mark.parametrize("ratios,index,expected", [([2, 3], 1, 0.5), ([1, 1, 2], 1, 2 / 3), ([2, 2, 2], 1, 0.5)])
def test_ks_jump_limits_and_ties(ratios, index, expected):
    for ordering in set(permutations(ratios)):
        assert _ks_tail_distance(np.array(ordering, dtype=float), index) == pytest.approx(expected)


@pytest.mark.parametrize("index", [0.3, 1, 2.5])
@pytest.mark.parametrize("ratios", [[1, 1, 1.5, 2, 4], [1.2, 1.2, 3, 9], [1, 2, 2, 2]])
def test_ks_against_independent_jump_enumeration_and_dense_grid(ratios, index):
    exact = _ks_tail_distance(np.array(ratios), index)
    assert exact == pytest.approx(independent_distance(ratios, index))
    # Include the left limit at each jump.
    grid = np.unique(
        np.concatenate(
            [np.linspace(1, max(ratios), 20001), ratios, [np.nextafter(x, -np.inf) for x in ratios if x > 1]]
        )
    )
    empirical = (np.array(ratios)[:, None] > grid).mean(axis=0)
    assert np.max(np.abs(empirical - grid ** (-index))) == pytest.approx(exact)


@pytest.mark.parametrize("degrees", [[9, 7, 3, 2, 1], [20, 10, 10, 5, 2, 2, 1, 0], [4, 4, 4, 2, 2, 1]])
def test_hill_selects_minimum_of_exact_suprema(degrees):
    ordered = sorted((x for x in degrees if x > 0), reverse=True)
    candidates = []
    for k in range(1, len(ordered)):
        ratios = [x / ordered[k] for x in ordered[:k]]
        log_mean = sum(np.log(x) for x in ratios) / k
        if log_mean > 0:
            index = 1 / log_mean
            candidates.append((independent_distance(ratios, index), k, index))
    distance, k, index = min(candidates, key=lambda entry: entry[0])
    actual = hill_tail_fit(degrees, min_tail=1, max_tail_fraction=1)
    assert actual.threshold_k == k
    assert actual.tail_index == pytest.approx(index)
    assert actual.ks_distance == pytest.approx(distance)
    assert actual.threshold_value == ordered[k]


def test_hill_uses_first_minimizer(monkeypatch):
    monkeypatch.setattr("model_fitting._ks_tail_distance", lambda *_: 0.25)
    assert hill_tail_fit([8, 4, 2, 1], min_tail=1, max_tail_fraction=1).threshold_k == 1


def test_hill_rejects_tied_constant_degrees():
    with pytest.raises(ValueError, match="Could not estimate"):
        hill_tail_fit([2, 2, 2], min_tail=1)
