"""Independent enumeration of batch synchronization and fixed-cut bounds."""

import pytest

from scripts.verify_gossip_rate_bound import recipients, verify_all


def test_direction_self_loop_and_recipient_deduplication():
    assert recipients((0, 1), {1}, "pull") == {0}
    assert recipients((0, 1), {1}, "push") == set()
    assert recipients((0, 1), {0}, "pp") == {1}
    assert recipients((0, 0), {0}, "pp") == set()
    assert recipients((0, 1), {0, 1}, "pp") == set()


def test_no_within_batch_cascade():
    snapshot = {0}
    result = recipients((1, 0), snapshot, "pull") | recipients((2, 1), snapshot, "pull")
    assert result == {1}
    assert snapshot == {0}


def test_all_three_vertex_graphs_single_session_bounds():
    report = verify_all((1,))
    assert report["passed"]
    assert report["graph_protocol_batch_cases"] == 180
    assert report["maximum_exact_mean_to_bound_ratio"] <= 1 + 1e-12


def test_empty_or_excessive_batch_specification_rejected():
    for batches in ((), (0,), (4,)):
        with pytest.raises(ValueError):
            verify_all(batches)
