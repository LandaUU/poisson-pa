from __future__ import annotations

import csv
import gzip
import io
import json
import os
from contextlib import redirect_stdout
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Literal, Optional, Sequence

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-master-deg-project")

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy import stats

from main import Graph, init_graph
from propagation import MessageRule, batch_recipients

HourMode = Literal["span", "active"]
PlotLanguage = Literal["en", "ru"]


@dataclass(slots=True, frozen=True)
class TemporalEdge:
    source: int
    target: int
    timestamp: int


@dataclass(slots=True, frozen=True)
class TimeAggregation:
    bucket_seconds: int
    first_timestamp: int
    last_timestamp: int
    first_bucket: int
    last_bucket: int
    span_buckets: int
    active_buckets: int
    edge_count: int


@dataclass(slots=True, frozen=True)
class TailFit:
    tail_index: float
    threshold_k: int
    threshold_value: float
    ks_distance: float
    observations: int


@dataclass(slots=True, frozen=True)
class PoissonPAFit:
    lambda_hat: float
    p_hat: float
    delta_in_hat: float
    delta_out_hat: float
    iota_in_hat: float
    iota_out_hat: float
    edge_count: int
    node_count: int
    new_node_count: int
    hour_mode: HourMode
    time_aggregation: TimeAggregation
    in_tail: TailFit
    out_tail: TailFit


@dataclass(slots=True)
class MessageCoverage:
    rule: MessageRule
    steps: np.ndarray
    nodes: np.ndarray
    informed: np.ndarray
    ratios: np.ndarray
    seed_node: int | None = None
    seed_selection: str = "first_event_source"


@dataclass(slots=True)
class SimulatedRun:
    run_id: int
    message_rule: MessageRule
    nodes: np.ndarray
    informed: np.ndarray
    ratios: np.ndarray
    in_degrees: np.ndarray
    out_degrees: np.ndarray


def read_temporal_edges(path: str | Path) -> list[TemporalEdge]:
    """Read SNAP-style temporal directed edges: SRC DST UNIXTS."""
    input_path = Path(path)
    if not input_path.exists():
        raise FileNotFoundError(f"CollegeMsg data file does not exist: {input_path}")

    open_fn = gzip.open if input_path.suffix == ".gz" else open
    edges: list[TemporalEdge] = []
    with open_fn(input_path, "rt", encoding="utf-8") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.replace(",", " ").split()
            if len(parts) < 3:
                raise ValueError(f"Expected at least three columns at {input_path}:{line_no}: {raw_line!r}")
            try:
                source, target, timestamp = (int(parts[0]), int(parts[1]), int(parts[2]))
            except ValueError as exc:
                raise ValueError(f"Invalid integer columns at {input_path}:{line_no}: {raw_line!r}") from exc
            edges.append(TemporalEdge(source=source, target=target, timestamp=timestamp))

    if not edges:
        raise ValueError(f"No temporal edges found in {input_path}")
    return sorted(edges, key=lambda edge: edge.timestamp)


def filter_edges(
    edges: Sequence[TemporalEdge],
    *,
    start_timestamp: Optional[int] = None,
    end_timestamp: Optional[int] = None,
) -> list[TemporalEdge]:
    filtered = [
        edge
        for edge in edges
        if (start_timestamp is None or edge.timestamp >= start_timestamp)
        and (end_timestamp is None or edge.timestamp <= end_timestamp)
    ]
    if not filtered:
        raise ValueError("No edges remain after timestamp filtering")
    return filtered


def build_multidigraph(edges: Sequence[TemporalEdge]) -> nx.MultiDiGraph:
    graph = nx.MultiDiGraph()
    graph.add_edges_from((edge.source, edge.target) for edge in edges)
    return graph


def degree_arrays_from_edges(edges: Sequence[TemporalEdge]) -> tuple[np.ndarray, np.ndarray, list[int]]:
    nodes = sorted({edge.source for edge in edges} | {edge.target for edge in edges})
    index = {node: idx for idx, node in enumerate(nodes)}
    in_degrees = np.zeros(len(nodes), dtype=np.float64)
    out_degrees = np.zeros(len(nodes), dtype=np.float64)
    for edge in edges:
        out_degrees[index[edge.source]] += 1.0
        in_degrees[index[edge.target]] += 1.0
    return in_degrees, out_degrees, nodes


def degree_arrays_from_graph(graph: nx.MultiDiGraph) -> tuple[np.ndarray, np.ndarray]:
    nodes = list(graph.nodes())
    in_degrees = np.fromiter((degree for _, degree in graph.in_degree(nodes)), dtype=np.float64, count=len(nodes))
    out_degrees = np.fromiter((degree for _, degree in graph.out_degree(nodes)), dtype=np.float64, count=len(nodes))
    return in_degrees, out_degrees


def time_aggregation(edges: Sequence[TemporalEdge], *, bucket_seconds: int = 3600) -> TimeAggregation:
    if bucket_seconds <= 0:
        raise ValueError("bucket_seconds must be positive")
    first_timestamp = min(edge.timestamp for edge in edges)
    last_timestamp = max(edge.timestamp for edge in edges)
    first_bucket = first_timestamp // bucket_seconds
    last_bucket = last_timestamp // bucket_seconds
    span_buckets = int(last_bucket - first_bucket + 1)
    active_buckets = len({edge.timestamp // bucket_seconds for edge in edges})
    return TimeAggregation(
        bucket_seconds=bucket_seconds,
        first_timestamp=first_timestamp,
        last_timestamp=last_timestamp,
        first_bucket=int(first_bucket),
        last_bucket=int(last_bucket),
        span_buckets=span_buckets,
        active_buckets=active_buckets,
        edge_count=len(edges),
    )


def count_new_node_appearances(edges: Sequence[TemporalEdge]) -> int:
    seen: set[int] = set()
    new_nodes = 0
    for edge in edges:
        if edge.source not in seen:
            new_nodes += 1
            seen.add(edge.source)
        if edge.target not in seen:
            new_nodes += 1
            seen.add(edge.target)
    return new_nodes


def _ks_tail_distance(ratios: np.ndarray, tail_index: float) -> float:
    y_values = np.unique(ratios)
    empirical = np.array([(ratios > y).mean() for y in y_values], dtype=np.float64)
    theoretical = np.power(y_values, -tail_index)
    return float(np.max(np.abs(empirical - theoretical)))


def hill_tail_fit(
    degrees: Sequence[float] | np.ndarray,
    *,
    min_tail: int = 10,
    max_tail_fraction: float = 0.5,
) -> TailFit:
    """Hill tail-index estimate with Clauset-style KS threshold selection."""
    values = np.asarray(degrees, dtype=np.float64)
    values = values[np.isfinite(values) & (values > 0)]
    values.sort()
    values = values[::-1]

    observations = int(values.size)
    if observations < 2:
        raise ValueError("Need at least two positive degree observations for Hill estimation")
    if not (0.0 < max_tail_fraction <= 1.0):
        raise ValueError("max_tail_fraction must be in (0, 1]")

    max_candidate = min(observations - 1, max(1, int(np.floor(observations * max_tail_fraction))))
    min_candidate = min(max(1, min_tail), max_candidate)

    best: TailFit | None = None
    for k in range(min_candidate, max_candidate + 1):
        threshold = float(values[k])
        if threshold <= 0.0:
            continue
        ratios = values[:k] / threshold
        log_mean = float(np.mean(np.log(ratios)))
        if log_mean <= 0.0 or not np.isfinite(log_mean):
            continue
        tail_index = 1.0 / log_mean
        ks_distance = _ks_tail_distance(ratios, tail_index)
        candidate = TailFit(
            tail_index=float(tail_index),
            threshold_k=int(k),
            threshold_value=threshold,
            ks_distance=ks_distance,
            observations=observations,
        )
        if best is None or candidate.ks_distance < best.ks_distance:
            best = candidate

    if best is None:
        raise ValueError("Could not estimate a positive Hill tail index")
    return best


def fit_poisson_pa_parameters(
    edges: Sequence[TemporalEdge],
    *,
    bucket_seconds: int = 3600,
    hour_mode: HourMode = "active",
    min_tail: int = 10,
    max_tail_fraction: float = 0.5,
) -> PoissonPAFit:
    if hour_mode not in {"span", "active"}:
        raise ValueError("hour_mode must be 'span' or 'active'")

    aggregation = time_aggregation(edges, bucket_seconds=bucket_seconds)
    denominator = aggregation.active_buckets if hour_mode == "active" else aggregation.span_buckets
    if denominator <= 0:
        raise ValueError("Non-positive time denominator for lambda estimate")

    in_degrees, out_degrees, nodes = degree_arrays_from_edges(edges)
    new_node_count = count_new_node_appearances(edges)
    p_hat = new_node_count / len(edges)
    if not (0.0 < p_hat < 1.0):
        raise ValueError(f"Estimated p must be inside (0, 1), got {p_hat:g}")

    in_tail = hill_tail_fit(in_degrees, min_tail=min_tail, max_tail_fraction=max_tail_fraction)
    out_tail = hill_tail_fit(out_degrees, min_tail=min_tail, max_tail_fraction=max_tail_fraction)
    iota_in = in_tail.tail_index
    iota_out = out_tail.tail_index
    delta_in = (iota_in - 1.0) / p_hat
    delta_out = (iota_out * (1.0 - p_hat) - 1.0) / p_hat

    return PoissonPAFit(
        lambda_hat=len(edges) / denominator,
        p_hat=float(p_hat),
        delta_in_hat=float(delta_in),
        delta_out_hat=float(delta_out),
        iota_in_hat=float(iota_in),
        iota_out_hat=float(iota_out),
        edge_count=len(edges),
        node_count=len(nodes),
        new_node_count=new_node_count,
        hour_mode=hour_mode,
        time_aggregation=aggregation,
        in_tail=in_tail,
        out_tail=out_tail,
    )


def constant_lambda(value: float) -> Callable[[int, float], float]:
    def _lambda(_step: int, _c: float) -> float:
        return float(value)

    return _lambda


def truncation_for_lambda(lambda_hat: float, *, minimum: int = 1_000) -> int:
    return max(minimum, int(np.ceil(lambda_hat + 12.0 * np.sqrt(max(lambda_hat, 1.0)) + 100.0)))


def _message_delta(
    *,
    graph: nx.MultiDiGraph,
    edges: Sequence[tuple[int, int]],
    informed: set[int],
    rule: MessageRule,
) -> int:
    recipients = batch_recipients(edges, informed, rule)
    informed.update(recipients)
    for node in recipients:
        graph.nodes[node]["message"] = True
    return len(recipients)


def simulate_fitted_model(
    fit: PoissonPAFit,
    *,
    runs: int,
    steps: Optional[int] = None,
    seed: int = 42,
    truncation_k: Optional[int] = None,
    message_rule: MessageRule = "target_to_source",
) -> list[SimulatedRun]:
    if fit.delta_in_hat <= 0.0 or fit.delta_out_hat <= 0.0:
        raise ValueError(
            "Fitted deltas must be positive to simulate the Wang-Resnick graph: "
            f"delta_in={fit.delta_in_hat:g}, delta_out={fit.delta_out_hat:g}"
        )
    if runs <= 0:
        raise ValueError("runs must be positive")

    if steps is None:
        steps = fit.time_aggregation.active_buckets if fit.hour_mode == "active" else fit.time_aggregation.span_buckets
    truncation_k = truncation_for_lambda(fit.lambda_hat) if truncation_k is None else truncation_k
    lambda_func = constant_lambda(fit.lambda_hat)
    simulated: list[SimulatedRun] = []

    for run_idx in range(runs):
        graph = Graph(
            c=0.0,
            p=fit.p_hat,
            delta_in=fit.delta_in_hat,
            delta_out=fit.delta_out_hat,
            lambda_func=lambda_func,
            rng=np.random.default_rng(seed + run_idx),
            message_rule=message_rule,
        ).initialize(init_graph(), initial_N=2, initial_S=1)

        with redirect_stdout(io.StringIO()):
            graph.evolute(rounds=steps, k=truncation_k)
        nodes, informed, ratios = graph.totals()
        in_degrees, out_degrees = degree_arrays_from_graph(graph._graph)
        simulated.append(
            SimulatedRun(
                run_id=run_idx + 1,
                message_rule=message_rule,
                nodes=nodes,
                informed=informed,
                ratios=ratios,
                in_degrees=in_degrees,
                out_degrees=out_degrees,
            )
        )

    return simulated


def temporal_message_coverage(
    edges: Sequence[TemporalEdge],
    *,
    rule: MessageRule = "target_to_source",
    bucket_seconds: int = 3600,
    initial_source: Optional[int] = None,
) -> MessageCoverage:
    if rule not in {"source_to_target", "target_to_source"}:
        raise ValueError("rule must be 'source_to_target' or 'target_to_source'")
    if not edges:
        raise ValueError("Need at least one edge for message coverage")

    ordered = sorted(edges, key=lambda edge: edge.timestamp)
    aggregation = time_aggregation(ordered, bucket_seconds=bucket_seconds)
    first_bucket = aggregation.first_bucket
    edges_by_bucket: list[list[TemporalEdge]] = [[] for _ in range(aggregation.span_buckets)]
    for edge in ordered:
        edges_by_bucket[int(edge.timestamp // bucket_seconds - first_bucket)].append(edge)

    informed = {ordered[0].source if initial_source is None else initial_source}
    seen: set[int] = set()
    nodes = np.zeros(aggregation.span_buckets, dtype=np.int64)
    informed_counts = np.zeros(aggregation.span_buckets, dtype=np.int64)

    for bucket_idx, bucket_edges in enumerate(edges_by_bucket):
        for edge in bucket_edges:
            seen.add(edge.source)
            seen.add(edge.target)
            informed.update(batch_recipients([(edge.source, edge.target)], informed, rule))
        nodes[bucket_idx] = len(seen)
        informed_counts[bucket_idx] = len(informed & seen)

    ratios = np.divide(
        informed_counts,
        nodes,
        out=np.zeros_like(informed_counts, dtype=np.float64),
        where=nodes > 0,
    )
    return MessageCoverage(
        rule=rule,
        steps=np.arange(aggregation.span_buckets, dtype=np.int64),
        nodes=nodes,
        informed=informed_counts,
        ratios=ratios,
        seed_node=ordered[0].source if initial_source is None else initial_source,
        seed_selection="first_event_source" if initial_source is None else "explicit_initial_source",
    )


def empirical_tail_survival(degrees: Sequence[float] | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(degrees, dtype=np.float64)
    values = values[np.isfinite(values) & (values > 0)]
    if values.size == 0:
        return np.empty(0), np.empty(0)
    unique = np.unique(values)
    survival = np.array([(values >= x).mean() for x in unique], dtype=np.float64)
    return unique, survival


def angular_samples(
    in_degrees: Sequence[float] | np.ndarray,
    out_degrees: Sequence[float] | np.ndarray,
    *,
    a: float,
    quantile: float = 0.995,
) -> np.ndarray:
    in_values = np.asarray(in_degrees, dtype=np.float64)
    out_values = np.asarray(out_degrees, dtype=np.float64)
    if in_values.shape != out_values.shape:
        raise ValueError("in_degrees and out_degrees must have the same shape")
    radius = in_values + out_values
    positive = radius > 0
    if not positive.any():
        return np.empty(0, dtype=np.float64)
    threshold = float(np.quantile(radius[positive], quantile))
    mask = positive & (radius >= threshold)
    if not mask.any():
        mask = radius == radius.max()
    in_power = np.power(in_values[mask], a)
    denominator = in_power + out_values[mask]
    return np.divide(in_power, denominator, out=np.zeros_like(in_power, dtype=np.float64), where=denominator > 0)


def localized_plot_path(output_path: str | Path, language: PlotLanguage) -> Path:
    path = Path(output_path)
    return path.with_name(f"{path.stem}_{language}{path.suffix}")


def _plot_text(language: PlotLanguage) -> dict[str, str]:
    if language == "ru":
        return {
            "simulated": "модель",
            "observed": "наблюдаемые данные",
            "in_degree_tail": "Хвост входящей степени",
            "out_degree_tail": "Хвост исходящей степени",
            "degree": "степень",
            "angular_title": "Сравнение угловой плотности",
            "density": "плотность",
            "coverage_title": "Сравнение охвата сообщением",
            "hour_bucket": "часовой интервал",
            "fitted_mean": "среднее fitted-модели",
            "real_degree_tail_overlay_title": "Сравнение хвостов степеней реальных датасетов",
            "out_tail_threshold_overlay_title": "Сравнение хвостовой области исходящей степени",
            "empirical_growth_title": "Рост сети и распространение сообщения",
            "relative_coverage_title": "Относительный охват",
            "number_of_nodes": "число узлов",
            "days_since_first_message": "дни с первого сообщения",
            "observed_nodes": r"наблюдаемые узлы $N_k$",
            "informed_nodes": r"информированные узлы $|S_k|$",
        }
    return {
        "simulated": "simulated",
        "observed": "observed",
        "in_degree_tail": "In-degree tail",
        "out_degree_tail": "Out-degree tail",
        "degree": "degree",
        "angular_title": "Angular density comparison",
        "density": "density",
        "coverage_title": "Message coverage comparison",
        "hour_bucket": "hour bucket",
        "fitted_mean": "fitted synthetic mean",
        "real_degree_tail_overlay_title": "Empirical degree-tail comparison",
        "out_tail_threshold_overlay_title": "Out-degree tail-region comparison",
        "empirical_growth_title": "Growth and information spread",
        "relative_coverage_title": "Relative coverage",
        "number_of_nodes": "number of nodes",
        "days_since_first_message": "days since first message",
        "observed_nodes": r"observed nodes $N_k$",
        "informed_nodes": r"informed nodes $|S_k|$",
    }


def _dataset_plot_label(dataset_label: str, language: PlotLanguage) -> str:
    if dataset_label == "observed":
        return _plot_text(language)["observed"]
    return dataset_label


def _density_on_unit_interval(samples: np.ndarray, grid: np.ndarray) -> np.ndarray:
    samples = _unit_interval_samples(samples)
    if samples.size < 2 or np.unique(samples).size < 2:
        if samples.size == 0:
            return np.zeros_like(grid)
        hist, edges = np.histogram(samples, bins=20, range=(0.0, 1.0), density=True)
        centers = (edges[:-1] + edges[1:]) / 2.0
        return np.interp(grid, centers, hist, left=0.0, right=0.0)
    kde = stats.gaussian_kde(samples)
    return np.clip(kde(grid), 0.0, None)


def _unit_interval_samples(samples: Sequence[float] | np.ndarray) -> np.ndarray:
    values = np.asarray(samples, dtype=np.float64)
    return values[np.isfinite(values) & (values >= 0.0) & (values <= 1.0)]


def _scott_bandwidth(samples: np.ndarray) -> float | None:
    if samples.size < 2 or np.unique(samples).size < 2:
        return None
    sample_std = float(np.std(samples, ddof=1))
    if not np.isfinite(sample_std) or sample_std <= 0.0:
        return None
    return sample_std * samples.size ** (-1.0 / 5.0)


def _gaussian_density_with_bandwidth(samples: np.ndarray, grid: np.ndarray, bandwidth: float) -> np.ndarray:
    if samples.size == 0 or not np.isfinite(bandwidth) or bandwidth <= 0.0:
        return np.zeros_like(grid)
    scaled = (grid[:, None] - samples[None, :]) / bandwidth
    density = np.exp(-0.5 * scaled * scaled).sum(axis=1)
    density /= samples.size * bandwidth * np.sqrt(2.0 * np.pi)
    return np.clip(density, 0.0, None)


def angular_bandwidth_diagnostic(
    samples: Sequence[float] | np.ndarray,
    *,
    label: str,
    bootstrap_samples: int = 200,
    grid_size: int = 512,
    seed: int = 20240521,
    multipliers: Sequence[float] | None = None,
) -> dict[str, object]:
    samples = _unit_interval_samples(samples)
    unique_count = int(np.unique(samples).size)
    result: dict[str, object] = {
        "label": label,
        "sample_count": int(samples.size),
        "unique_count": unique_count,
        "bootstrap_samples": int(bootstrap_samples),
        "grid_size": int(grid_size),
        "seed": int(seed),
        "interval": [0.0, 1.0],
        "boundary_correction": "none",
    }

    scott_bandwidth = _scott_bandwidth(samples)
    if scott_bandwidth is None:
        result["status"] = "insufficient variation"
        result["scott_bandwidth"] = None
        result["bootstrap_bandwidth"] = None
        result["bandwidth_multiplier"] = None
        result["ise_curve"] = []
        return result

    multiplier_values = np.asarray(
        multipliers if multipliers is not None else np.linspace(0.3, 2.0, 25), dtype=np.float64
    )
    multiplier_values = multiplier_values[np.isfinite(multiplier_values) & (multiplier_values > 0.0)]
    if multiplier_values.size == 0:
        raise ValueError("At least one positive finite bandwidth multiplier is required")

    grid = np.linspace(0.0, 1.0, grid_size)
    pilot_density = _gaussian_density_with_bandwidth(samples, grid, scott_bandwidth)
    ise_sum = np.zeros(multiplier_values.size, dtype=np.float64)
    rng = np.random.default_rng(seed)
    for _bootstrap_index in range(bootstrap_samples):
        resampled = samples[rng.integers(0, samples.size, size=samples.size)]
        for multiplier_index, multiplier in enumerate(multiplier_values):
            density = _gaussian_density_with_bandwidth(resampled, grid, scott_bandwidth * float(multiplier))
            difference = density - pilot_density
            ise_sum[multiplier_index] += float(np.trapezoid(difference * difference, grid))

    mean_ise = ise_sum / bootstrap_samples
    best_index = int(np.argmin(mean_ise))
    bandwidth_multiplier = float(multiplier_values[best_index])
    bootstrap_bandwidth = scott_bandwidth * bandwidth_multiplier
    bootstrap_density = _gaussian_density_with_bandwidth(samples, grid, bootstrap_bandwidth)
    density_difference = bootstrap_density - pilot_density
    density_ise = float(np.trapezoid(density_difference * density_difference, grid))

    result.update(
        {
            "status": "ok",
            "scott_bandwidth": float(scott_bandwidth),
            "bootstrap_bandwidth": float(bootstrap_bandwidth),
            "bandwidth_multiplier": bandwidth_multiplier,
            "selected_on_grid_boundary": bool(best_index == 0 or best_index == multiplier_values.size - 1),
            "scott_vs_bootstrap_density_ise": density_ise,
            "ise_curve": [
                {
                    "multiplier": float(multiplier),
                    "bandwidth": float(scott_bandwidth * multiplier),
                    "bootstrap_mean_ise": float(ise),
                }
                for multiplier, ise in zip(multiplier_values, mean_ise)
            ],
        }
    )
    return result


def plot_degree_tail_comparison(
    *,
    real_in_degrees: np.ndarray,
    real_out_degrees: np.ndarray,
    simulated_runs: Sequence[SimulatedRun],
    output_path: str | Path,
    dataset_label: str = "observed",
    language: PlotLanguage = "en",
    max_simulated_lines: int = 20,
) -> None:
    labels = _plot_text(language)
    real_label = _dataset_plot_label(dataset_label, language)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
    panels = [
        (
            axes[0],
            real_in_degrees,
            [run.in_degrees for run in simulated_runs],
            labels["in_degree_tail"],
            r"$P(I \geq x)$",
        ),
        (
            axes[1],
            real_out_degrees,
            [run.out_degrees for run in simulated_runs],
            labels["out_degree_tail"],
            r"$P(O \geq x)$",
        ),
    ]
    for ax, real_values, simulated_values, title, ylabel in panels:
        for idx, values in enumerate(simulated_values[:max_simulated_lines]):
            x, y = empirical_tail_survival(values)
            if x.size:
                ax.loglog(
                    x, y, color="tab:red", alpha=0.18, linewidth=0.9, label=labels["simulated"] if idx == 0 else None
                )
        x_real, y_real = empirical_tail_survival(real_values)
        if x_real.size:
            ax.loglog(x_real, y_real, "o-", color="black", markersize=3, linewidth=1.1, label=real_label)
        ax.set_title(title)
        ax.set_xlabel(labels["degree"])
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25, which="both")
        ax.legend()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_real_dataset_degree_tail_overlay(
    *,
    datasets: Sequence[tuple[str, np.ndarray, np.ndarray]],
    output_path: str | Path,
    language: PlotLanguage = "en",
) -> None:
    labels = _plot_text(language)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), layout="constrained")
    panels = [
        (axes[0], 1, labels["in_degree_tail"], r"$P(I \geq x)$"),
        (axes[1], 2, labels["out_degree_tail"], r"$P(O \geq x)$"),
    ]
    for ax, degree_idx, title, ylabel in panels:
        for dataset_idx, dataset in enumerate(datasets):
            dataset_label = dataset[0]
            values = dataset[degree_idx]
            x, y = empirical_tail_survival(values)
            if x.size:
                ax.loglog(
                    x,
                    y,
                    "o-",
                    color=colors[dataset_idx % len(colors)],
                    markersize=2.4,
                    linewidth=1.15,
                    alpha=0.9,
                    label=dataset_label,
                )
        ax.set_title(title)
        ax.set_xlabel(labels["degree"])
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25, which="both")
        ax.legend()
    fig.suptitle(labels["real_degree_tail_overlay_title"])
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_real_dataset_out_tail_threshold_overlay(
    *,
    datasets: Sequence[tuple[str, np.ndarray, float, float]],
    output_path: str | Path,
    language: PlotLanguage = "en",
) -> None:
    labels = _plot_text(language)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    fig, ax = plt.subplots(figsize=(8.8, 5.2), layout="constrained")
    for dataset_idx, (dataset_label, out_degrees, threshold, tail_index) in enumerate(datasets):
        values = np.asarray(out_degrees, dtype=np.float64)
        values = values[np.isfinite(values) & (values >= threshold)]
        if values.size == 0 or threshold <= 0:
            continue
        x = np.unique(values)
        y = np.array([(values >= value).mean() for value in x], dtype=np.float64)
        legend_label = rf"{dataset_label}, $\hat{{\iota}}_{{out}}={tail_index:.3f}$, $\tau_{{out}}={threshold:g}$"
        ax.loglog(
            x / threshold,
            y,
            "o-",
            color=colors[dataset_idx % len(colors)],
            markersize=2.6,
            linewidth=1.3,
            alpha=0.92,
            label=legend_label,
        )
    ax.set_title(labels["out_tail_threshold_overlay_title"])
    ax.set_xlabel(r"$O/\tau_{out}$")
    ax.set_ylabel(r"$P(O\geq x\mid O\geq \tau_{out})$")
    ax.grid(alpha=0.25, which="both")
    ax.legend()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_angular_density_comparison(
    *,
    real_angles: np.ndarray,
    simulated_angles: Sequence[np.ndarray],
    output_path: str | Path,
    dataset_label: str = "observed",
    language: PlotLanguage = "en",
) -> None:
    labels = _plot_text(language)
    real_label = _dataset_plot_label(dataset_label, language)
    grid = np.linspace(0.0, 1.0, 250)
    fig, ax = plt.subplots(figsize=(7.5, 4.5), layout="constrained")
    if simulated_angles:
        pooled = np.concatenate([angles for angles in simulated_angles if angles.size])
        if pooled.size:
            ax.plot(
                grid, _density_on_unit_interval(pooled, grid), color="tab:red", linewidth=1.8, label=labels["simulated"]
            )
    if real_angles.size:
        ax.plot(grid, _density_on_unit_interval(real_angles, grid), color="black", linewidth=1.8, label=real_label)
    ax.set_xlabel(r"$\theta$")
    ax.set_ylabel(labels["density"])
    ax.set_title(labels["angular_title"])
    ax.grid(alpha=0.25)
    ax.legend()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_message_coverage_comparison(
    *,
    real_coverage: MessageCoverage,
    simulated_runs: Sequence[SimulatedRun],
    output_path: str | Path,
    language: PlotLanguage = "en",
) -> None:
    labels = _plot_text(language)
    observed_label = f"{labels['observed']} ({real_coverage.rule})"
    fig, ax = plt.subplots(figsize=(8.5, 4.8), layout="constrained")
    if simulated_runs:
        max_len = max(run.ratios.size for run in simulated_runs)
        padded = np.full((len(simulated_runs), max_len), np.nan, dtype=np.float64)
        for idx, run in enumerate(simulated_runs):
            padded[idx, : run.ratios.size] = run.ratios
        mean = np.nanmean(padded, axis=0)
        std = np.nanstd(padded, axis=0, ddof=1) if len(simulated_runs) > 1 else np.zeros_like(mean)
        x = np.arange(max_len)
        ax.plot(x, mean, color="tab:red", linewidth=1.8, label=labels["fitted_mean"])
        ax.fill_between(x, np.clip(mean - std, 0.0, 1.0), np.clip(mean + std, 0.0, 1.0), color="tab:red", alpha=0.15)
    ax.plot(real_coverage.steps, real_coverage.ratios, color="black", linewidth=1.5, label=observed_label)
    ax.set_xlabel(labels["hour_bucket"])
    ax.set_ylabel(r"$|S_k| / N_k$")
    ax.set_title(labels["coverage_title"])
    ax.grid(alpha=0.25)
    ax.legend()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_empirical_message_dynamics(
    *,
    real_coverage: MessageCoverage,
    output_path: str | Path,
    language: PlotLanguage = "en",
) -> None:
    labels = _plot_text(language)
    fig, axes = plt.subplots(1, 2, figsize=(12.35, 4.8), layout="constrained")

    axes[0].plot(
        real_coverage.steps, real_coverage.nodes, color="tab:blue", linewidth=2.2, label=labels["observed_nodes"]
    )
    axes[0].plot(
        real_coverage.steps, real_coverage.informed, color="#d04a02", linewidth=2.2, label=labels["informed_nodes"]
    )
    axes[0].set_title(labels["empirical_growth_title"])
    axes[0].set_xlabel(labels["days_since_first_message"])
    axes[0].set_ylabel(labels["number_of_nodes"])
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].plot(real_coverage.steps, real_coverage.ratios, color="#2f7d44", linewidth=2.2)
    axes[1].set_title(labels["relative_coverage_title"])
    axes[1].set_xlabel(labels["days_since_first_message"])
    axes[1].set_ylabel(r"$|S_k| / N_k$")
    axes[1].grid(alpha=0.25)

    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_real_coverage_csv(coverages: Sequence[MessageCoverage], output_path: str | Path) -> None:
    with Path(output_path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["rule", "step", "N_obs", "S_obs", "ratio"])
        for coverage in coverages:
            for step, nodes, informed, ratio in zip(coverage.steps, coverage.nodes, coverage.informed, coverage.ratios):
                writer.writerow([coverage.rule, int(step), int(nodes), int(informed), float(ratio)])


def write_simulated_trajectories_csv(simulated_runs: Sequence[SimulatedRun], output_path: str | Path) -> None:
    with Path(output_path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["run_id", "message_rule", "step", "N_k", "S_k", "ratio"])
        for run in simulated_runs:
            for step, nodes, informed, ratio in zip(range(run.nodes.size), run.nodes, run.informed, run.ratios):
                writer.writerow([run.run_id, run.message_rule, int(step), int(nodes), int(informed), float(ratio)])


def fit_to_json_dict(
    *,
    fit: PoissonPAFit,
    selected_coverage: MessageCoverage,
    all_coverages: Sequence[MessageCoverage],
    simulated_runs: Sequence[SimulatedRun],
    config: dict[str, object],
    skipped_simulation_reason: Optional[str] = None,
) -> dict[str, object]:
    coverage_summary = {
        coverage.rule: {
            "final_nodes": int(coverage.nodes[-1]),
            "final_informed": int(coverage.informed[-1]),
            "final_ratio": float(coverage.ratios[-1]),
        }
        for coverage in all_coverages
    }
    simulation_summary: dict[str, object]
    if simulated_runs:
        final_nodes = np.array([run.nodes[-1] for run in simulated_runs], dtype=np.float64)
        final_informed = np.array([run.informed[-1] for run in simulated_runs], dtype=np.float64)
        final_ratios = np.array([run.ratios[-1] for run in simulated_runs], dtype=np.float64)
        simulation_summary = {
            "runs": len(simulated_runs),
            "message_rule": simulated_runs[0].message_rule,
            "final_nodes_mean": float(final_nodes.mean()),
            "final_informed_mean": float(final_informed.mean()),
            "final_ratio_mean": float(final_ratios.mean()),
            "final_ratio_std": float(final_ratios.std(ddof=1)) if len(simulated_runs) > 1 else 0.0,
        }
    else:
        simulation_summary = {"runs": 0, "skipped_reason": skipped_simulation_reason}

    return {
        "config": config,
        "fit": asdict(fit),
        "selected_message_rule": selected_coverage.rule,
        "propagation": {
            "real_direction": selected_coverage.rule,
            "synthetic_direction": simulated_runs[0].message_rule
            if simulated_runs
            else config.get("synthetic_message_rule"),
            "synthetic_update": "batch_start_snapshot",
            "temporal_update": "one_event_at_a_time",
            "equal_timestamp_order": "stable_input_order",
            "active_contacts": "new_edges_only",
            "real_seed_selection": selected_coverage.seed_selection,
            "real_seed_node": selected_coverage.seed_node,
            "synthetic_seed_selection": config.get("synthetic_seed_selection", "initial_graph_node_0"),
            "synthetic_seed_node": config.get("synthetic_seed_node", 0),
        },
        "real_message_coverage": coverage_summary,
        "simulation": simulation_summary,
    }


def write_fit_summary(summary: dict[str, object], output_path: str | Path) -> None:
    Path(output_path).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
