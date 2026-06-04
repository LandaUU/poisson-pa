#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from dataclasses import asdict
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-master-deg-project")

import matplotlib
import numpy as np
from matplotlib import pyplot as plt

from scripts.run_sk_kstar_experiment import ExperimentConfig, run_experiment

matplotlib.use("Agg")


def parse_float_list(raw: str) -> list[float]:
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def read_single_row_csv(path: Path) -> dict[str, str]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    if len(rows) != 1:
        raise ValueError(f"Expected exactly one row in {path}, got {len(rows)}")
    return rows[0]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
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
    threshold = float(grid.mean())
    for row_index in range(grid.shape[0]):
        for col_index in range(grid.shape[1]):
            value = grid[row_index, col_index]
            ax.text(
                col_index,
                row_index,
                f"{value:.3f}",
                ha="center",
                va="center",
                color="white" if value > threshold else "black",
                fontsize=8.5,
            )


def draw_heatmap(
    *,
    ax: plt.Axes,
    fig: plt.Figure,
    grid: np.ndarray,
    p_values: list[float],
    c_values: list[float],
    title: str,
    colorbar_label: str,
) -> None:
    image = ax.imshow(grid, cmap="YlGnBu", aspect="auto")
    ax.set_xticks(range(len(c_values)), [f"{value:g}" for value in c_values])
    ax.set_yticks(range(len(p_values)), [f"{value:g}" for value in p_values])
    ax.set_xlabel("c")
    ax.set_ylabel("p")
    ax.set_title(title)
    annotate_heatmap(ax, grid)
    colorbar = fig.colorbar(image, ax=ax, shrink=0.88)
    colorbar.set_label(colorbar_label)


def write_combined_summary(
    *,
    batch_dir: Path,
    summary_rows: list[tuple[ExperimentConfig, Path, dict[str, str]]],
) -> None:
    output_path = batch_dir / "combined_kstar_summary.csv"
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "p",
                "c",
                "runs",
                "nu",
                "t_star",
                "poisson_mean",
                "max_K_star",
                "mean_K_star",
                "std_K_star",
                "q25_K_star",
                "q50_K_star",
                "q75_K_star",
                "mean_N_K_star",
                "std_N_K_star",
                "mean_S_K_star",
                "std_S_K_star",
                "mean_ratio",
                "std_ratio",
                "ratio_q05",
                "ratio_q25",
                "ratio_q50",
                "ratio_q75",
                "ratio_q95",
                "experiment_dir",
            ]
        )

        for config, experiment_dir, row in summary_rows:
            writer.writerow(
                [
                    config.p,
                    config.c,
                    config.runs,
                    config.nu,
                    config.t_star,
                    row["poisson_mean"],
                    row["max_K_star"],
                    row["mean_K_star"],
                    row["std_K_star"],
                    row["q25_K_star"],
                    row["q50_K_star"],
                    row["q75_K_star"],
                    row["mean_N_K_star"],
                    row["std_N_K_star"],
                    row["mean_S_K_star"],
                    row["std_S_K_star"],
                    row["mean_ratio"],
                    row["std_ratio"],
                    row["ratio_q05"],
                    row["ratio_q25"],
                    row["ratio_q50"],
                    row["ratio_q75"],
                    row["ratio_q95"],
                    str(experiment_dir),
                ]
            )


def write_combined_thresholds(
    *,
    batch_dir: Path,
    threshold_rows: list[tuple[ExperimentConfig, Path, list[dict[str, str]]]],
) -> None:
    output_path = batch_dir / "combined_thresholds.csv"
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "p",
                "c",
                "runs",
                "nu",
                "t_star",
                "theta",
                "count_ge_theta",
                "prob_ge_theta",
                "experiment_dir",
            ]
        )
        for config, experiment_dir, rows in threshold_rows:
            for row in rows:
                writer.writerow(
                    [
                        config.p,
                        config.c,
                        config.runs,
                        config.nu,
                        config.t_star,
                        row["theta"],
                        row["count_ge_theta"],
                        row["prob_ge_theta"],
                        str(experiment_dir),
                    ]
                )


def plot_summary_heatmaps(
    *,
    batch_dir: Path,
    combined_rows: list[dict[str, str]],
    p_values: list[float],
    c_values: list[float],
) -> None:
    mean_s_grid = build_metric_grid(
        rows=combined_rows,
        p_values=p_values,
        c_values=c_values,
        field="mean_S_K_star",
    )
    std_s_grid = build_metric_grid(
        rows=combined_rows,
        p_values=p_values,
        c_values=c_values,
        field="std_S_K_star",
    )
    mean_ratio_grid = build_metric_grid(
        rows=combined_rows,
        p_values=p_values,
        c_values=c_values,
        field="mean_ratio",
    )
    std_ratio_grid = build_metric_grid(
        rows=combined_rows,
        p_values=p_values,
        c_values=c_values,
        field="std_ratio",
    )

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), layout="constrained")
    draw_heatmap(
        ax=axes[0, 0],
        fig=fig,
        grid=mean_s_grid,
        p_values=p_values,
        c_values=c_values,
        title=r"Среднее значение $|S_{K^{*}}|$",
        colorbar_label=r"$\overline{|S_{K^{*}}|}$",
    )
    draw_heatmap(
        ax=axes[0, 1],
        fig=fig,
        grid=std_s_grid,
        p_values=p_values,
        c_values=c_values,
        title=r"Стандартное отклонение $|S_{K^{*}}|$",
        colorbar_label=r"$\mathrm{std}(|S_{K^{*}}|)$",
    )
    draw_heatmap(
        ax=axes[1, 0],
        fig=fig,
        grid=mean_ratio_grid,
        p_values=p_values,
        c_values=c_values,
        title=r"Среднее значение $|S_{K^{*}}| / N_{K^{*}}$",
        colorbar_label=r"$\overline{|S_{K^{*}}| / N_{K^{*}}}$",
    )
    draw_heatmap(
        ax=axes[1, 1],
        fig=fig,
        grid=std_ratio_grid,
        p_values=p_values,
        c_values=c_values,
        title=r"Стандартное отклонение $|S_{K^{*}}| / N_{K^{*}}$",
        colorbar_label=r"$\mathrm{std}(|S_{K^{*}}| / N_{K^{*}})$",
    )
    fig.suptitle(r"Сводка для $|S_{K^{*}}|$ и $|S_{K^{*}}| / N_{K^{*}}$")
    fig.savefig(batch_dir / "kstar_summary_heatmaps.png", dpi=180)
    plt.close(fig)


def plot_threshold_heatmaps(
    *,
    batch_dir: Path,
    threshold_rows: list[dict[str, str]],
    p_values: list[float],
    c_values: list[float],
) -> None:
    theta_values = sorted({float(row["theta"]) for row in threshold_rows})
    if not theta_values:
        return

    rows_count = math.ceil(len(theta_values) / 2)
    fig, axes = plt.subplots(rows_count, 2, figsize=(10, 4 * rows_count), layout="constrained")
    axes_array = np.atleast_1d(axes).reshape(rows_count, 2)

    for axis in axes_array.ravel()[len(theta_values) :]:
        axis.axis("off")

    for index, theta in enumerate(theta_values):
        filtered_rows = [row for row in threshold_rows if float(row["theta"]) == theta]
        grid = build_metric_grid(
            rows=filtered_rows,
            p_values=p_values,
            c_values=c_values,
            field="prob_ge_theta",
        )
        axis = axes_array.ravel()[index]
        draw_heatmap(
            ax=axis,
            fig=fig,
            grid=grid,
            p_values=p_values,
            c_values=c_values,
            title=rf"$P(|S_{{K^*}}| / N_{{K^*}} \geq {theta:g})$",
            colorbar_label="Probability",
        )

    fig.suptitle(r"Вероятности превышения порога для $|S_{K^{*}}| / N_{K^{*}}$")
    fig.savefig(batch_dir / "kstar_threshold_heatmaps.png", dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a grid of Monte Carlo experiments for |S_{K*}| and |S_{K*}| / N_{K*} with K* ~ Poisson(nu * T*)."
        )
    )
    parser.add_argument("--p-values", default="0.1,0.5,0.9")
    parser.add_argument("--c-values", default="0.1,0.2,0.3")
    parser.add_argument("--runs", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--delta-in", type=int, default=1)
    parser.add_argument("--delta-out", type=int, default=1)
    parser.add_argument("--truncation-k", type=int, default=10_000)
    parser.add_argument("--output-root", default="results/sk_kstar/grid")
    parser.add_argument("--nu", type=float, default=1.0)
    parser.add_argument("--t-star", type=float, default=200.0)
    parser.add_argument("--theta-values", default="0.7,0.8,0.9,0.95")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    p_values = parse_float_list(args.p_values)
    c_values = parse_float_list(args.c_values)
    theta_values = tuple(parse_float_list(args.theta_values))

    batch_dir = Path(args.output_root) / (
        f"grid_runs_{args.runs}_nu_{args.nu:g}_t_{args.t_star:g}"
        f"_p_{'-'.join(f'{value:g}' for value in p_values)}"
        f"_c_{'-'.join(f'{value:g}' for value in c_values)}"
    )
    batch_dir.mkdir(parents=True, exist_ok=True)

    batch_config = {
        "p_values": p_values,
        "c_values": c_values,
        "runs": args.runs,
        "seed": args.seed,
        "delta_in": args.delta_in,
        "delta_out": args.delta_out,
        "truncation_k": args.truncation_k,
        "nu": args.nu,
        "t_star": args.t_star,
        "theta_values": theta_values,
    }
    (batch_dir / "grid_config.json").write_text(
        json.dumps(batch_config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    summary_rows: list[tuple[ExperimentConfig, Path, dict[str, str]]] = []
    threshold_rows: list[tuple[ExperimentConfig, Path, list[dict[str, str]]]] = []

    for p_index, p_value in enumerate(p_values):
        for c_index, c_value in enumerate(c_values):
            config = ExperimentConfig(
                p=p_value,
                c=c_value,
                runs=args.runs,
                seed=args.seed + 10_000 * p_index + 100 * c_index,
                delta_in=args.delta_in,
                delta_out=args.delta_out,
                truncation_k=args.truncation_k,
                output_dir=str(batch_dir),
                nu=args.nu,
                t_star=args.t_star,
                theta_values=theta_values,
            )
            experiment_dir = run_experiment(config)
            summary_row = read_single_row_csv(experiment_dir / "kstar_summary.csv")
            threshold_row_set = read_rows(experiment_dir / "coverage_thresholds.csv")
            summary_rows.append((config, experiment_dir, summary_row))
            threshold_rows.append((config, experiment_dir, threshold_row_set))
            print(f"Completed K* experiment p={config.p:g}, c={config.c:g} -> {experiment_dir}")

    write_combined_summary(batch_dir=batch_dir, summary_rows=summary_rows)
    write_combined_thresholds(batch_dir=batch_dir, threshold_rows=threshold_rows)

    combined_rows = read_rows(batch_dir / "combined_kstar_summary.csv")
    combined_threshold_rows = read_rows(batch_dir / "combined_thresholds.csv")
    plot_summary_heatmaps(
        batch_dir=batch_dir,
        combined_rows=combined_rows,
        p_values=p_values,
        c_values=c_values,
    )
    plot_threshold_heatmaps(
        batch_dir=batch_dir,
        threshold_rows=combined_threshold_rows,
        p_values=p_values,
        c_values=c_values,
    )

    manifest = [
        {
            **asdict(config),
            "experiment_dir": str(experiment_dir),
        }
        for config, experiment_dir, _summary in summary_rows
    ]
    (batch_dir / "experiment_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Saved K* grid artifacts to: {batch_dir}")


if __name__ == "__main__":
    main()
