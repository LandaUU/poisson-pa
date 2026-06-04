#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import asdict
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-master-deg-project")

import matplotlib
import numpy as np
from matplotlib import pyplot as plt

from scripts.run_sk_experiment import ExperimentConfig, run_experiment

matplotlib.use("Agg")


def parse_float_list(raw: str) -> list[float]:
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def parse_int_list(raw: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in raw.split(",") if item.strip())


def read_summary_rows(summary_path: Path) -> list[dict[str, str]]:
    with summary_path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def build_metric_grid(
    *,
    rows: list[dict[str, str]],
    p_values: list[float],
    c_values: list[float],
    field: str,
) -> np.ndarray:
    lookup = {(float(row["p"]), float(row["c"])): float(row[field]) for row in rows}
    grid = np.empty((len(p_values), len(c_values)), dtype=np.float64)
    for p_index, p_value in enumerate(p_values):
        for c_index, c_value in enumerate(c_values):
            grid[p_index, c_index] = lookup[(p_value, c_value)]
    return grid


def annotate_heatmap(ax: plt.Axes, grid: np.ndarray) -> None:
    for row_index in range(grid.shape[0]):
        for col_index in range(grid.shape[1]):
            value = grid[row_index, col_index]
            text_color = "white" if value > grid.mean() else "black"
            ax.text(
                col_index,
                row_index,
                f"{value:.3f}",
                ha="center",
                va="center",
                color=text_color,
                fontsize=8.5,
            )


def plot_heatmap_pair(
    *,
    batch_dir: Path,
    p_values: list[float],
    c_values: list[float],
    left_grid: np.ndarray,
    left_title: str,
    left_format: str,
    right_grid: np.ndarray,
    right_title: str,
    right_format: str,
    figure_title: str,
    output_name: str,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), layout="constrained")
    for ax, grid, title, colorbar_label in (
        (axes[0], left_grid, left_title, left_format),
        (axes[1], right_grid, right_title, right_format),
    ):
        image = ax.imshow(grid, cmap="YlGnBu", aspect="auto")
        ax.set_xticks(range(len(c_values)), [f"{value:g}" for value in c_values])
        ax.set_yticks(range(len(p_values)), [f"{value:g}" for value in p_values])
        ax.set_xlabel("c")
        ax.set_ylabel("p")
        ax.set_title(title)
        annotate_heatmap(ax, grid)
        colorbar = fig.colorbar(image, ax=ax, shrink=0.88)
        colorbar.set_label(colorbar_label)

    fig.suptitle(figure_title)
    fig.savefig(batch_dir / output_name, dpi=180)
    plt.close(fig)


def write_combined_summary(
    *,
    batch_dir: Path,
    experiment_rows: list[tuple[ExperimentConfig, Path, list[dict[str, str]]]],
) -> None:
    output_path = batch_dir / "combined_sk_summary.csv"
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "p",
                "c",
                "runs",
                "steps",
                "step",
                "mean_N_k",
                "std_N_k",
                "mean_S_k",
                "std_S_k",
                "mean_ratio",
                "std_ratio",
                "ratio_q25",
                "ratio_q50",
                "ratio_q75",
                "experiment_dir",
            ]
        )

        for config, experiment_dir, rows in experiment_rows:
            for row in rows:
                writer.writerow(
                    [
                        config.p,
                        config.c,
                        config.runs,
                        config.steps,
                        row["step"],
                        row["mean_N_k"],
                        row["std_N_k"],
                        row["mean_S_k"],
                        row["std_S_k"],
                        row["mean_ratio"],
                        row["std_ratio"],
                        row["ratio_q25"],
                        row["ratio_q50"],
                        row["ratio_q75"],
                        str(experiment_dir),
                    ]
                )


def write_final_step_summary(
    *,
    batch_dir: Path,
    experiment_rows: list[tuple[ExperimentConfig, Path, list[dict[str, str]]]],
) -> None:
    output_path = batch_dir / "final_step_summary.csv"
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "p",
                "c",
                "runs",
                "steps",
                "final_mean_N_k",
                "final_std_N_k",
                "final_mean_S_k",
                "final_std_S_k",
                "final_mean_ratio",
                "final_std_ratio",
                "final_ratio_q25",
                "final_ratio_q50",
                "final_ratio_q75",
                "experiment_dir",
            ]
        )

        for config, experiment_dir, rows in experiment_rows:
            final_row = rows[-1]
            writer.writerow(
                [
                    config.p,
                    config.c,
                    config.runs,
                    config.steps,
                    final_row["mean_N_k"],
                    final_row["std_N_k"],
                    final_row["mean_S_k"],
                    final_row["std_S_k"],
                    final_row["mean_ratio"],
                    final_row["std_ratio"],
                    final_row["ratio_q25"],
                    final_row["ratio_q50"],
                    final_row["ratio_q75"],
                    str(experiment_dir),
                ]
            )


def plot_final_heatmaps(
    *,
    batch_dir: Path,
    final_rows: list[dict[str, str]],
    p_values: list[float],
    c_values: list[float],
) -> None:
    mean_s_grid = build_metric_grid(
        rows=final_rows,
        p_values=p_values,
        c_values=c_values,
        field="final_mean_S_k",
    )
    std_s_grid = build_metric_grid(
        rows=final_rows,
        p_values=p_values,
        c_values=c_values,
        field="final_std_S_k",
    )
    mean_ratio_grid = build_metric_grid(
        rows=final_rows,
        p_values=p_values,
        c_values=c_values,
        field="final_mean_ratio",
    )
    std_ratio_grid = build_metric_grid(
        rows=final_rows,
        p_values=p_values,
        c_values=c_values,
        field="final_std_ratio",
    )

    plot_heatmap_pair(
        batch_dir=batch_dir,
        p_values=p_values,
        c_values=c_values,
        left_grid=mean_s_grid,
        left_title="Mean of #S_k",
        left_format="mean(#S_k)",
        right_grid=std_s_grid,
        right_title="Standard deviation of #S_k",
        right_format="std(#S_k)",
        figure_title="#S_k at the final step over the parameter grid (p,c)",
        output_name="sk_heatmaps.png",
    )
    plot_heatmap_pair(
        batch_dir=batch_dir,
        p_values=p_values,
        c_values=c_values,
        left_grid=mean_ratio_grid,
        left_title="Mean of #S_k / N_k",
        left_format="mean(#S_k / N_k)",
        right_grid=std_ratio_grid,
        right_title="Standard deviation of #S_k / N_k",
        right_format="std(#S_k / N_k)",
        figure_title="#S_k / N_k at the final step over the parameter grid (p,c)",
        output_name="sk_ratio_heatmaps.png",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=("Run a grid of S_k Monte Carlo experiments and aggregate their summaries.")
    )
    parser.add_argument("--p-values", default="0.1,0.5,0.9")
    parser.add_argument("--c-values", default="0.1,0.2,0.3")
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--delta-in", type=int, default=1)
    parser.add_argument("--delta-out", type=int, default=1)
    parser.add_argument("--truncation-k", type=int, default=10_000)
    parser.add_argument("--output-root", default="results/sk/grid")
    parser.add_argument(
        "--boxplot-steps",
        default="20,100,200",
        help="Comma-separated list of k values for boxplots.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    p_values = parse_float_list(args.p_values)
    c_values = parse_float_list(args.c_values)
    boxplot_steps = parse_int_list(args.boxplot_steps)

    batch_dir = Path(args.output_root) / (
        f"grid_runs_{args.runs}_steps_{args.steps}"
        f"_p_{'-'.join(f'{value:g}' for value in p_values)}"
        f"_c_{'-'.join(f'{value:g}' for value in c_values)}"
    )
    batch_dir.mkdir(parents=True, exist_ok=True)

    batch_config = {
        "p_values": p_values,
        "c_values": c_values,
        "runs": args.runs,
        "steps": args.steps,
        "seed": args.seed,
        "delta_in": args.delta_in,
        "delta_out": args.delta_out,
        "truncation_k": args.truncation_k,
        "boxplot_steps": boxplot_steps,
    }
    (batch_dir / "grid_config.json").write_text(
        json.dumps(batch_config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    experiment_rows: list[tuple[ExperimentConfig, Path, list[dict[str, str]]]] = []

    for p_index, p_value in enumerate(p_values):
        for c_index, c_value in enumerate(c_values):
            config = ExperimentConfig(
                p=p_value,
                c=c_value,
                runs=args.runs,
                steps=args.steps,
                seed=args.seed + 10_000 * p_index + 100 * c_index,
                delta_in=args.delta_in,
                delta_out=args.delta_out,
                truncation_k=args.truncation_k,
                output_dir=str(batch_dir),
                boxplot_steps=boxplot_steps,
            )
            experiment_dir = run_experiment(config)
            summary_rows = read_summary_rows(experiment_dir / "sk_summary.csv")
            experiment_rows.append((config, experiment_dir, summary_rows))
            print(f"Completed experiment p={config.p:g}, c={config.c:g} -> {experiment_dir}")

    write_combined_summary(batch_dir=batch_dir, experiment_rows=experiment_rows)
    write_final_step_summary(batch_dir=batch_dir, experiment_rows=experiment_rows)
    final_rows = read_summary_rows(batch_dir / "final_step_summary.csv")
    plot_final_heatmaps(
        batch_dir=batch_dir,
        final_rows=final_rows,
        p_values=p_values,
        c_values=c_values,
    )

    manifest = [
        {
            **asdict(config),
            "experiment_dir": str(experiment_dir),
        }
        for config, experiment_dir, _rows in experiment_rows
    ]
    (batch_dir / "experiment_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Saved grid artifacts to: {batch_dir}")


if __name__ == "__main__":
    main()
