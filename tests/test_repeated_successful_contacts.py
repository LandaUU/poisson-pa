import math

import pytest

from scripts.repeated_successful_contacts import (
    contact_metrics,
    exact_categories,
    old_hits_pmf,
)


def test_distinguish_recipient_repeats_from_parallel_edges_and_no_cascade():
    # Node 1 has two informed targets; node 2 has parallel contacts.
    # Node 3 stays uninformed because the batch cannot cascade.
    edges = [(1, 0), (1, 5), (2, 0), (2, 0), (3, 1), (4, 0), (0, 5)]
    result = contact_metrics(edges, {0, 1, 2, 3, 5}, {0, 5})
    assert result == {
        "successful_contacts": 5,
        "distinct_newly_informed": 3,
        "old_successful_contacts": 4,
        "old_distinct_recipients": 2,
        "old_multihit_recipients": 2,
        "redundant_successful_contacts": 2,
        "repeated_successful_pairs": 1,
        "redundant_parallel_contacts": 1,
    }


def test_triple_hit_counts_two_extra_contacts_but_one_recipient():
    result = contact_metrics([(1, 0)] * 3, {0, 1}, {0})
    assert result["redundant_successful_contacts"] == 2
    assert result["old_multihit_recipients"] == 1


@pytest.mark.parametrize("q", [0, 0.125, 0.25, 1])
def test_exact_law_matches_explicit_fixed_batch_enumeration(q):
    intensity = 0.5
    for r in range(6):
        expected = 0.0
        for m in range(1, 24):
            if r <= m:
                expected += (
                    math.exp(-intensity)
                    * intensity ** (m - 1)
                    / math.factorial(m - 1)
                    * math.comb(m, r)
                    * q**r
                    * (1 - q) ** (m - r)
                )
        assert old_hits_pmf(r, q, intensity) == pytest.approx(expected, abs=1e-14)
    for probabilities in exact_categories(q, intensity).values():
        assert min(probabilities) >= 0
        assert sum(probabilities) == pytest.approx(1)
