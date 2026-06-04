#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-master-deg-project")

import matplotlib
import numpy as np
from matplotlib import pyplot as plt

from main import Graph, init_graph

matplotlib.use("Agg")


@dataclass(slots=True)
class ExperimentConfig:
    p_values: tuple[float, ...] = (0.1, 0.5, 0.9)
    c_values: tuple[float, ...] = (0.1, 0.2, 0.3)
    snapshot_steps: tuple[int, ...] = (10, 30, 50, 100)
    base_runs: int = 5
    replications: int = 500
    seed: int = 42
    delta_in: int = 1
    delta_out: int = 1
    truncation_k: int = 10_000
    output_dir: str = "results/pn/g0_1"


@dataclass(slots=True)
class SnapshotRow:
    p: float
    c: float
    base_run: int
    step: int
    nodes: int
    informed: int
    a_n: float
    p_theory: float
    p_empirical: float
    abs_error: float
    successes: int
    replications: int


def parse_float_list(raw: str) -> tuple[float, ...]:
    return tuple(float(item.strip()) for item in raw.split(",") if item.strip())


def parse_int_list(raw: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in raw.split(",") if item.strip())


def parse_args() -> ExperimentConfig:
    parser = argparse.ArgumentParser(
        description=("Run a local p_n experiment for the initial state G0^(1): two nodes with self-loops.")
    )
    parser.add_argument("--p-values", default="0.1,0.5,0.9")
    parser.add_argument("--c-values", default="0.1,0.2,0.3")
    parser.add_argument("--snapshot-steps", default="10,30,50,100")
    parser.add_argument("--base-runs", type=int, default=5)
    parser.add_argument("--replications", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--delta-in", type=int, default=1)
    parser.add_argument("--delta-out", type=int, default=1)
    parser.add_argument("--truncation-k", type=int, default=10_000)
    parser.add_argument("--output-dir", default="results/pn/g0_1")
    args = parser.parse_args()

    return ExperimentConfig(
        p_values=parse_float_list(args.p_values),
        c_values=parse_float_list(args.c_values),
        snapshot_steps=parse_int_list(args.snapshot_steps),
        base_runs=args.base_runs,
        replications=args.replications,
        seed=args.seed,
        delta_in=args.delta_in,
        delta_out=args.delta_out,
        truncation_k=args.truncation_k,
        output_dir=args.output_dir,
    )


def make_graph(*, p: float, c: float, config: ExperimentConfig, seed: int) -> Graph:
    return Graph(
        c=c,
        p=p,
        delta_in=config.delta_in,
        delta_out=config.delta_out,
        rng=np.random.default_rng(seed),
    ).initialize(init_graph(), initial_N=2, initial_S=1)


def informed_nodes(graph) -> set[int]:
    return {node for node, data in graph.nodes(data=True) if data.get("message", False)}


def compute_a_n(graph, *, informed: set[int], p: float, delta_in: int, delta_out: int) -> float:
    nodes = list(graph.nodes())
    sum_in_s = sum(graph.in_degree(node) + delta_in for node in informed)
    sum_in_all = sum(graph.in_degree(node) + delta_in for node in nodes)
    sum_out_non_s = sum(graph.out_degree(node) + delta_out for node in nodes if node not in informed)
    sum_out_all = sum(graph.out_degree(node) + delta_out for node in nodes)

    target_informed_prob = sum_in_s / sum_in_all
    source_uninformed_prob = sum_out_non_s / sum_out_all

    return p * target_informed_prob + (1.0 - p) * source_uninformed_prob * target_informed_prob


def compute_theoretical_p_n(
    graph_state: Graph,
    *,
    step: int,
    informed: set[int],
) -> tuple[float, float]:
    a_n = compute_a_n(
        graph_state._graph,
        informed=informed,
        p=graph_state.params.p,
        delta_in=graph_state.params.delta_in,
        delta_out=graph_state.params.delta_out,
    )
    lambda_value = float(graph_state.lambda_func(step + 1, graph_state.params.c))
    p_theory = 1.0 - (1.0 - a_n) * np.exp(-lambda_value * a_n)
    return a_n, float(p_theory)


def clone_sampler_at_snapshot(source: Graph, *, seed: int) -> Graph:
    sampler = Graph(
        c=source.params.c,
        p=source.params.p,
        delta_in=source.params.delta_in,
        delta_out=source.params.delta_out,
        lambda_func=source.lambda_func,
        rng=np.random.default_rng(seed),
    ).initialize(
        initial_state=source._graph.copy(),
        initial_N=int(source.N_total[-1]),
        initial_S=int(source.S_total[-1]),
    )
    return sampler


def empirical_p_n(snapshot: Graph, *, step: int, config: ExperimentConfig, seed: int) -> tuple[int, float]:
    sampler = clone_sampler_at_snapshot(snapshot, seed=seed)
    informed = informed_nodes(snapshot._graph)
    successes = 0

    for _ in range(config.replications):
        edges, _new_nodes = sampler.sample_step_edges(
            current_round=step + 1,
            k=config.truncation_k,
        )
        success = any((source not in informed) and (target in informed) for source, target in edges)
        if success:
            successes += 1

    return successes, successes / config.replications


def build_snapshots(config: ExperimentConfig) -> list[SnapshotRow]:
    rows: list[SnapshotRow] = []
    max_step = max(config.snapshot_steps)

    for p_index, p_value in enumerate(config.p_values):
        for c_index, c_value in enumerate(config.c_values):
            for base_run in range(1, config.base_runs + 1):
                base_seed = config.seed + 100_000 * p_index + 1_000 * c_index + base_run
                graph = make_graph(p=p_value, c=c_value, config=config, seed=base_seed)
                target_steps = set(config.snapshot_steps)

                for current_step in range(1, max_step + 1):
                    with contextlib.redirect_stdout(io.StringIO()):
                        graph.evolute(rounds=1, k=config.truncation_k)

                    if current_step not in target_steps:
                        continue

                    informed = informed_nodes(graph._graph)
                    a_n, p_theory = compute_theoretical_p_n(
                        graph,
                        step=current_step,
                        informed=informed,
                    )
                    replay_seed = base_seed * 10_000 + current_step
                    successes, p_empirical = empirical_p_n(
                        graph,
                        step=current_step,
                        config=config,
                        seed=replay_seed,
                    )

                    rows.append(
                        SnapshotRow(
                            p=p_value,
                            c=c_value,
                            base_run=base_run,
                            step=current_step,
                            nodes=int(graph.N_total[-1]),
                            informed=int(graph.S_total[-1]),
                            a_n=a_n,
                            p_theory=p_theory,
                            p_empirical=p_empirical,
                            abs_error=abs(p_empirical - p_theory),
                            successes=successes,
                            replications=config.replications,
                        )
                    )

    return rows


def save_raw_rows(rows: list[SnapshotRow], *, output_dir: Path) -> None:
    csv_path = output_dir / "pn_snapshot_rows.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "p",
                "c",
                "base_run",
                "step",
                "nodes",
                "informed",
                "a_n",
                "p_theory",
                "p_empirical",
                "abs_error",
                "successes",
                "replications",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row.p,
                    row.c,
                    row.base_run,
                    row.step,
                    row.nodes,
                    row.informed,
                    row.a_n,
                    row.p_theory,
                    row.p_empirical,
                    row.abs_error,
                    row.successes,
                    row.replications,
                ]
            )


def grouped_summary(rows: list[SnapshotRow]) -> list[dict[str, float]]:
    grouped: dict[tuple[float, float, int], list[SnapshotRow]] = {}
    for row in rows:
        grouped.setdefault((row.p, row.c, row.step), []).append(row)

    summary: list[dict[str, float]] = []
    for (p_value, c_value, step), group_rows in sorted(grouped.items()):
        theory = np.array([row.p_theory for row in group_rows], dtype=np.float64)
        empirical = np.array([row.p_empirical for row in group_rows], dtype=np.float64)
        errors = np.array([row.abs_error for row in group_rows], dtype=np.float64)
        summary.append(
            {
                "p": p_value,
                "c": c_value,
                "step": step,
                "mean_p_theory": float(theory.mean()),
                "mean_p_empirical": float(empirical.mean()),
                "mean_abs_error": float(errors.mean()),
                "max_abs_error": float(errors.max()),
                "std_p_empirical": float(empirical.std(ddof=1)) if len(empirical) > 1 else 0.0,
                "states": float(len(group_rows)),
            }
        )
    return summary


def save_summary(rows: list[dict[str, float]], *, output_dir: Path) -> None:
    csv_path = output_dir / "pn_summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "p",
                "c",
                "step",
                "mean_p_theory",
                "mean_p_empirical",
                "mean_abs_error",
                "max_abs_error",
                "std_p_empirical",
                "states",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row["p"],
                    row["c"],
                    row["step"],
                    row["mean_p_theory"],
                    row["mean_p_empirical"],
                    row["mean_abs_error"],
                    row["max_abs_error"],
                    row["std_p_empirical"],
                    int(row["states"]),
                ]
            )


def plot_theory_vs_empirical(rows: list[SnapshotRow], *, output_dir: Path) -> None:
    theory = np.array([row.p_theory for row in rows], dtype=np.float64)
    empirical = np.array([row.p_empirical for row in rows], dtype=np.float64)

    fig, ax = plt.subplots(figsize=(5.5, 5.5), layout="constrained")
    ax.scatter(theory, empirical, s=18, alpha=0.7, color="tab:blue")
    diagonal = np.linspace(0.0, 1.0, 200)
    ax.plot(diagonal, diagonal, color="black", linewidth=1.2, linestyle="--")
    ax.set_xlabel(r"Theoretical value of $p_n$")
    ax.set_ylabel(r"Empirical estimate $\widehat{p}_n$")
    ax.set_title(r"Comparison of theoretical and empirical values of $p_n$")
    ax.grid(alpha=0.25)
    fig.savefig(output_dir / "pn_theory_vs_empirical.png", dpi=180)
    plt.close(fig)


def save_config(config: ExperimentConfig, *, output_dir: Path) -> None:
    (output_dir / "config.json").write_text(
        json.dumps(asdict(config), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def run_experiment(config: ExperimentConfig) -> Path:
    output_dir = Path(config.output_dir) / (
        f"base_runs_{config.base_runs}_repl_{config.replications}"
        f"_p_{'-'.join(f'{value:g}' for value in config.p_values)}"
        f"_c_{'-'.join(f'{value:g}' for value in config.c_values)}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = build_snapshots(config)
    summary = grouped_summary(rows)

    save_config(config, output_dir=output_dir)
    save_raw_rows(rows, output_dir=output_dir)
    save_summary(summary, output_dir=output_dir)
    plot_theory_vs_empirical(rows, output_dir=output_dir)

    return output_dir


def main() -> None:
    config = parse_args()
    output_dir = run_experiment(config)
    print(f"Saved p_n experiment artifacts to: {output_dir}")


if __name__ == "__main__":
    main()
