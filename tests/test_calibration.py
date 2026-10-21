from dataclasses import replace

import numpy as np
import pytest

import model_fitting
from main import Graph
from model_fitting import (
    TailFit,
    TemporalEdge,
    constant_lambda,
    diagnose_growth,
    edge_count_calibration,
    fit_poisson_pa_parameters,
    fit_to_json_dict,
    simulate_fitted_model,
    temporal_message_coverage,
)


@pytest.fixture
def introductions():
    return [
        TemporalEdge(0, 1, 0),
        TemporalEdge(2, 0, 0),
        TemporalEdge(0, 3, 3600),
        TemporalEdge(4, 5, 3600),
        TemporalEdge(6, 6, 7200),
        TemporalEdge(0, 1, 7200),
    ]


def test_growth_classes_and_initial_endpoints(introductions):
    diagnostic = diagnose_growth(introductions)
    assert diagnostic.initial_nodes == (0, 1)
    assert diagnostic.existing_endpoints == 2
    assert diagnostic.new_source_only == 1
    assert diagnostic.new_target_only == 1
    assert diagnostic.both_new == 1
    assert diagnostic.new_self_loop == 1
    assert diagnostic.new_node_count == 5
    assert diagnostic.batch_violating_events == 3


def test_initial_selfloop_and_stable_order():
    edges = [TemporalEdge(0, 0, 0), TemporalEdge(2, 1, 3600), TemporalEdge(1, 0, 3600)]
    diagnostic = diagnose_growth(edges)
    assert diagnostic.initial_nodes == (0,)
    assert diagnostic.both_new == 1
    assert diagnostic.new_node_count == 2
    assert diagnostic.batch_violating_events == 2


def test_reuse_of_new_sources_is_batch_violation():
    edges = [
        TemporalEdge(0, 1, 0),
        TemporalEdge(2, 0, 1),
        TemporalEdge(2, 1, 2),
        TemporalEdge(3, 2, 3),
        TemporalEdge(2, 0, 3600),
    ]
    diagnostic = diagnose_growth(edges)
    assert diagnostic.new_source_only == 2
    assert diagnostic.batch_violating_events == 2
    assert diagnose_growth(edges, bucket_seconds=1).batch_violating_events == 0


@pytest.mark.parametrize("mode", ["active", "span"])
def test_shifted_mean(introductions, mode):
    aggregation, mean, parameter = edge_count_calibration(introductions, hour_mode=mode)
    assert aggregation.active_buckets == 3
    assert mean == 2
    assert parameter == 1


def test_empty_calendar_intervals():
    edges = [TemporalEdge(0, 1, 0), TemporalEdge(1, 0, 7200)]
    assert edge_count_calibration(edges)[2] == 0
    with pytest.raises(ValueError, match="Empty intervals"):
        edge_count_calibration(edges, hour_mode="span")


@pytest.mark.parametrize("function", [diagnose_growth, edge_count_calibration, fit_poisson_pa_parameters])
def test_no_events(function):
    with pytest.raises(ValueError, match="at least one"):
        function([])


@pytest.mark.parametrize("lambda_value", [-1, np.inf, np.nan])
def test_invalid_lambda(lambda_value):
    graph = Graph(0.2, 0.5, 1, 1, lambda_func=constant_lambda(lambda_value)).initialize()
    with pytest.raises(ValueError, match="finite and nonnegative"):
        graph.sample_delta_m(1, 10)


def test_lambda_zero_and_truncation():
    graph = Graph(0.2, 0.5, 1, 1, lambda_func=constant_lambda(0)).initialize()
    assert [graph.sample_delta_m(step, 10) for step in range(1, 6)] == [1] * 5
    with pytest.raises(ValueError, match="positive"):
        graph.sample_delta_m(1, 0)


def test_calibration_moments_and_metadata(introductions, monkeypatch):
    monkeypatch.setattr(model_fitting, "hill_tail_fit", lambda *args, **kwargs: TailFit(10.0, 2, 1.0, 0.1, 7))
    fit = fit_poisson_pa_parameters(introductions)
    assert fit.new_node_count == 5  # Neither 7 observed nodes nor 4 introducing events.
    assert fit.p_hat == pytest.approx(5 / 6)
    assert fit.lambda_hat == 1
    assert fit.mean_edges_per_step == 2
    assert fit.initial_node_count == 2
    assert fit.calibration_method == "effective_moment_matching"
    assert 6 * fit.p_hat == 5
    coverage = temporal_message_coverage(introductions)
    report = fit_to_json_dict(
        fit=fit, selected_coverage=coverage, all_coverages=[coverage], simulated_runs=[], config={}
    )
    assert report["fit"]["growth_diagnostics"]["new_target_only"] == 1
    assert report["calibration"]["limitations"]


def test_fitted_initial_count_and_zero_lambda(fit):
    run = simulate_fitted_model(replace(fit, initial_node_count=1, lambda_hat=0), runs=1, steps=5)[0]
    assert run.nodes[0] == run.informed[0] == 1
    assert np.sum(run.in_degrees) == 6  # One initial loop, then five contacts.
    assert np.all((run.informed >= 1) & (run.informed <= run.nodes))
