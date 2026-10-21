from dataclasses import replace

import networkx as nx
import numpy as np
import pytest

import model_fitting
from model_fitting import (
    TailFit,
    TemporalEdge,
    diagnose_growth,
    fit_poisson_pa_parameters,
    simulate_fitted_model,
    temporal_message_coverage,
)


def test_replay_counts_initial_population_without_rescanning_old_edges():
    # Old 1->0 and 2->1 contacts belong to G0; only new events transmit.
    events = [TemporalEdge(2, 2, 0), TemporalEdge(1, 0, 1), TemporalEdge(2, 1, 2)]
    replay = temporal_message_coverage(events, initial_nodes=[0, 1, 2, 3], initial_source=0)
    assert replay.nodes.tolist() == [4]
    assert replay.informed.tolist() == [3]
    no_new_contact = temporal_message_coverage(events[:1], initial_nodes=[0, 1, 2, 3], initial_source=0)
    assert no_new_contact.informed.tolist() == [1]


def test_growth_uses_all_prefix_vertices():
    events = [TemporalEdge(2, 0, 0), TemporalEdge(3, 1, 1)]
    growth = diagnose_growth(events, initial_nodes=[0, 1, 2])
    assert growth.existing_endpoints == 1
    assert growth.new_source_only == growth.new_node_count == 1
    assert growth.initial_nodes == (0, 1, 2)


def test_fit_uses_remainder_counts_but_supplied_full_graph_for_tails(monkeypatch):
    captured = []

    def hill(degrees, **kwargs):
        captured.append(np.asarray(degrees))
        return TailFit(3.0, 1, 1.0, 0.1, len(degrees))

    monkeypatch.setattr(model_fitting, "hill_tail_fit", hill)
    prefix = [TemporalEdge(0, 1, 0), TemporalEdge(1, 2, 1)]
    remaining = [TemporalEdge(3, 0, 3600), TemporalEdge(1, 0, 3601)]
    fit = fit_poisson_pa_parameters(remaining, initial_nodes=[0, 1, 2], tail_edges=prefix + remaining)
    assert fit.initial_node_count == 3 and fit.node_count == 4
    assert fit.new_node_count == 1 and fit.p_hat == 0.5
    assert fit.edge_count == 2 and fit.lambda_hat == 1
    assert captured[0].sum() == captured[1].sum() == 4


def test_fitted_copies_prefix_graph_and_uses_only_selected_seed(fit):
    graph = nx.MultiDiGraph([(0, 1), (0, 1), (1, 2), (2, 0)])
    graph.nodes[0]["message"] = True
    fitted = replace(fit, initial_node_count=3)
    runs = simulate_fitted_model(fitted, runs=2, steps=2, initial_state=graph, initial_informed=[2])
    assert all(run.nodes[0] == 3 and run.informed[0] == 1 for run in runs)
    zero = simulate_fitted_model(fitted, runs=1, steps=0, initial_state=graph, initial_informed=[2])[0]
    assert zero.in_degrees.tolist() == [1, 2, 1]
    assert zero.out_degrees.tolist() == [2, 1, 1]
    assert graph.number_of_edges() == 4
    assert graph.nodes[0]["message"] is True
    assert not graph.nodes[2].get("message", False)


@pytest.mark.parametrize("ids,seed", [([1, 2, 3], [1]), ([0, 1, 2], [3]), ([0, 1, 2], [])])
def test_invalid_fitted_initialization(fit, ids, seed):
    graph = nx.MultiDiGraph()
    graph.add_nodes_from(ids)
    with pytest.raises(ValueError):
        simulate_fitted_model(
            replace(fit, initial_node_count=3), runs=1, steps=1, initial_state=graph, initial_informed=seed
        )


def test_replay_seed_must_be_in_initial_graph():
    with pytest.raises(ValueError, match="seed"):
        temporal_message_coverage([TemporalEdge(2, 0, 0)], initial_nodes=[0, 1], initial_source=2)
