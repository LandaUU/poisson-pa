import itertools
import json
import sys

import networkx as nx
import numpy as np
import pytest

from main import Graph, init_graph
from model_fitting import (
    TemporalEdge,
    _message_delta,
    constant_lambda,
    fit_to_json_dict,
    read_temporal_edges,
    simulate_fitted_model,
    temporal_message_coverage,
    write_real_coverage_csv,
    write_simulated_trajectories_csv,
)
from propagation import batch_recipients
from scripts.run_collegemsg_fit import CollegeMsgFitConfig, parse_args


@pytest.mark.parametrize(
    "edges,expected",
    [
        ([(1, 0)], {1}),
        ([(0, 1)], set()),
        ([(1, 0), (1, 0)], {1}),
        ([(2, 0)], {2}),
        ([(1, 1)], set()),
    ],
)
def test_pull_contacts(edges, expected):
    snapshot = {0}
    assert batch_recipients(edges, snapshot) == expected
    assert snapshot == {0}


@pytest.mark.parametrize(
    "rule,edges",
    [
        ("target_to_source", [(1, 0), (2, 1)]),
        ("source_to_target", [(0, 1), (1, 2)]),
    ],
)
@pytest.mark.parametrize("reverse", [False, True])
def test_snapshot_both_paths(rule, edges, reverse):
    contacts = edges[::-1] if reverse else edges
    graph = Graph(0.2, 0.5, 1, 1, message_rule=rule).initialize()
    graph._graph.add_edges_from(contacts)
    graph._simple_message_distribution(graph._graph, contacts)
    assert graph.S[-1] == 1
    assert {n for n, attrs in graph._graph.nodes(data=True) if attrs.get("message")} == {0, 1}
    fitted_graph = nx.MultiDiGraph(contacts)
    informed = {0}
    assert _message_delta(graph=fitted_graph, edges=contacts, informed=informed, rule=rule) == 1
    assert informed == {0, 1}
    next_contact = edges[1:]
    assert _message_delta(graph=fitted_graph, edges=next_contact, informed=informed, rule=rule) == 1
    assert informed == {0, 1, 2}


def test_all_permutations():
    for edges in itertools.permutations([(1, 0), (2, 1), (1, 0)]):
        assert batch_recipients(edges, {0}) == {1}


def test_only_new_edges_and_new_source(monkeypatch):
    graph = Graph(0.2, 0.5, 1, 1).initialize()
    graph._graph.add_edge(1, 0)  # An old contact must not transmit.
    graph.N_total[0] = 2
    batches = iter([([(1, 1)], 0), ([(2, 0), (2, 0)], 1)])
    monkeypatch.setattr(graph, "sample_step_edges", lambda **kwargs: next(batches))
    graph.evolute(2, k=20)
    assert graph.S == [1, 0, 1]
    assert not graph._graph.nodes[1].get("message", False)
    assert graph._graph.nodes[2]["message"]
    assert graph.S_total == [1, 1, 2]


@pytest.mark.parametrize("rule", ["target_to_source", "source_to_target"])
def test_fitted_matches_main_and_reproducible(fit, rule):
    runs = simulate_fitted_model(fit, runs=2, steps=20, seed=42, truncation_k=1000, message_rule=rule)
    repeated = simulate_fitted_model(fit, runs=2, steps=20, seed=42, truncation_k=1000, message_rule=rule)
    for idx, (run, again) in enumerate(zip(runs, repeated)):
        graph = Graph(
            0, 0.5, 1, 1, lambda_func=constant_lambda(2), rng=np.random.default_rng(42 + idx), message_rule=rule
        ).initialize(init_graph(), 2, 1)
        graph.evolute(20, k=1000)
        nodes, informed, ratios = graph.totals()
        for actual, expected in zip((run.nodes, run.informed, run.ratios), (nodes, informed, ratios)):
            np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(run.informed, again.informed)
        np.testing.assert_array_equal(run.in_degrees, again.in_degrees)
        assert np.all((run.informed >= 1) & (run.informed <= run.nodes))


@pytest.mark.parametrize("same_time", [False, True])
def test_temporal_sequence_and_first_source(same_time):
    times = [0, 0, 0] if same_time else [0, 1, 2]
    edges = [TemporalEdge(0, 9, times[0]), TemporalEdge(1, 0, times[1]), TemporalEdge(2, 1, times[2])]
    coverage = temporal_message_coverage(edges)
    assert coverage.informed.tolist() == [3]
    assert coverage.nodes.tolist() == [4]
    assert coverage.seed_node == 0
    assert coverage.seed_selection == "first_event_source"
    reversed_chain = [edges[0], edges[2], edges[1]]
    if same_time:
        assert temporal_message_coverage(reversed_chain).informed.tolist() == [2]
    assert temporal_message_coverage(edges[:1]).informed.tolist() == [1]


def test_stable_read_and_hour_statistics(tmp_path):
    path = tmp_path / "contacts.txt"
    path.write_text("2 1 3600\n0 9 0\n1 0 3600\n")
    edges = read_temporal_edges(path)
    assert [(edge.source, edge.target) for edge in edges] == [(0, 9), (2, 1), (1, 0)]
    assert temporal_message_coverage(edges).informed.tolist() == [1, 2]
    assert temporal_message_coverage(edges, bucket_seconds=7200).informed.tolist() == [2]


def test_cli_defaults(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["fit", "--data-path", "example.txt"])
    config = parse_args()
    assert config.real_message_rule == config.synthetic_message_rule == "target_to_source"
    assert CollegeMsgFitConfig("example.txt").real_message_rule == "target_to_source"
    monkeypatch.setattr(
        sys, "argv", ["fit", "--data-path", "example.txt", "--synthetic-message-rule", "source_to_target"]
    )
    assert parse_args().synthetic_message_rule == "source_to_target"


def test_report_and_csv_schemas(fit, tmp_path):
    coverage = temporal_message_coverage([TemporalEdge(0, 1, 0)])
    runs = simulate_fitted_model(fit, runs=1, steps=2)
    report = fit_to_json_dict(
        fit=fit, selected_coverage=coverage, all_coverages=[coverage], simulated_runs=runs, config={}
    )
    assert report["propagation"]["synthetic_update"] == "batch_start_snapshot"
    assert report["propagation"]["real_seed_node"] == 0
    assert report["propagation"]["real_direction"] == "target_to_source"
    assert {"config", "fit", "selected_message_rule", "real_message_coverage", "simulation"} <= report.keys()
    (tmp_path / "report.json").write_text(json.dumps(report))
    write_real_coverage_csv([coverage], tmp_path / "real.csv")
    write_simulated_trajectories_csv(runs, tmp_path / "synthetic.csv")
    assert (tmp_path / "synthetic.csv").read_text().splitlines()[0] == "run_id,message_rule,step,N_k,S_k,ratio"
    assert (tmp_path / "real.csv").read_text().splitlines()[0] == "rule,step,N_obs,S_obs,ratio"


def test_invalid_rule():
    with pytest.raises(ValueError):
        batch_recipients([], set(), "invalid")


@pytest.mark.parametrize(
    "rule,first,second",
    [
        ("target_to_source", [(1, 0), (2, 1)], [(2, 1)]),
        ("source_to_target", [(0, 1), (1, 2)], [(1, 2)]),
    ],
)
def test_fitted_controlled_batches(fit, monkeypatch, rule, first, second):
    def sample(self, current_round, k):
        return (first, 1) if current_round == 1 else (second, 0)

    monkeypatch.setattr(Graph, "sample_step_edges", sample)
    run = simulate_fitted_model(fit, runs=1, steps=2, message_rule=rule)[0]
    assert run.nodes.tolist() == [2, 3, 3]
    assert run.informed.tolist() == [1, 2, 3]


def test_explicit_temporal_push_and_seed_override():
    edges = [TemporalEdge(0, 1, 0), TemporalEdge(1, 2, 0)]
    assert temporal_message_coverage(edges, rule="source_to_target").informed.tolist() == [3]
    coverage = temporal_message_coverage(edges, initial_source=1)
    assert coverage.informed.tolist() == [2]
    assert coverage.seed_node == 1
    assert coverage.seed_selection == "explicit_initial_source"
