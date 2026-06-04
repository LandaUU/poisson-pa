#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-master-deg-project")

import matplotlib
import numpy as np
from matplotlib import pyplot as plt

from main import collect_trajectories_over_runs, make_default_graph

matplotlib.use("Agg")


@dataclass(slots=True)
class ExperimentConfig:
    p: float = 0.5
    c: float = 0.2
    runs: int = 100
    steps: int = 200
    seed: int = 42
    delta_in: int = 1
    delta_out: int = 1
    truncation_k: int = 10_000
    output_dir: str = "results/sk"
    boxplot_steps: tuple[int, ...] = (20, 100, 200)


def parse_int_list(raw: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in raw.split(",") if item.strip())


def parse_args() -> ExperimentConfig:
    parser = argparse.ArgumentParser(
        description=(
            "Run an ensemble Monte Carlo experiment for #S_k and #S_k / N_k "
            "using the same model implementation as main.py."
        )
    )
    parser.add_argument("--p", type=float, default=0.5)
    parser.add_argument("--c", type=float, default=0.2)
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--delta-in", type=int, default=1)
    parser.add_argument("--delta-out", type=int, default=1)
    parser.add_argument("--truncation-k", type=int, default=10_000)
    parser.add_argument("--output-dir", default="results/sk")
    parser.add_argument("--boxplot-steps", default="20,100,200")
    args = parser.parse_args()

    return ExperimentConfig(
        p=args.p,
        c=args.c,
        runs=args.runs,
        steps=args.steps,
        seed=args.seed,
        delta_in=args.delta_in,
        delta_out=args.delta_out,
        truncation_k=args.truncation_k,
        output_dir=args.output_dir,
        boxplot_steps=parse_int_list(args.boxplot_steps),
    )


def simulate_trajectories(config: ExperimentConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return collect_trajectories_over_runs(
        make_graph=lambda run_idx: make_default_graph(
            c=config.c,
            p=config.p,
            delta_in=config.delta_in,
            delta_out=config.delta_out,
            rng=np.random.default_rng(config.seed + run_idx),
        ),
        evolute_steps=config.steps,
        k=config.truncation_k,
        runs=config.runs,
    )


def save_trajectories_csv(
    *,
    n_trajectories: np.ndarray,
    s_trajectories: np.ndarray,
    ratios: np.ndarray,
    output_dir: Path,
) -> None:
    csv_path = output_dir / "sk_trajectories.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["run_id", "step", "N_k", "S_k", "ratio"])
        for run_id in range(n_trajectories.shape[0]):
            for step in range(n_trajectories.shape[1]):
                writer.writerow(
                    [
                        run_id + 1,
                        step,
                        int(n_trajectories[run_id, step]),
                        int(s_trajectories[run_id, step]),
                        float(ratios[run_id, step]),
                    ]
                )


def save_summary_csv(
    *,
    n_trajectories: np.ndarray,
    s_trajectories: np.ndarray,
    ratios: np.ndarray,
    output_dir: Path,
) -> None:
    csv_path = output_dir / "sk_summary.csv"
    n_mean = n_trajectories.mean(axis=0)
    n_std = n_trajectories.std(axis=0, ddof=1)
    s_mean = s_trajectories.mean(axis=0)
    s_std = s_trajectories.std(axis=0, ddof=1)
    ratio_mean = ratios.mean(axis=0)
    ratio_std = ratios.std(axis=0, ddof=1)
    ratio_q25 = np.quantile(ratios, 0.25, axis=0)
    ratio_q50 = np.quantile(ratios, 0.50, axis=0)
    ratio_q75 = np.quantile(ratios, 0.75, axis=0)

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
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
            ]
        )
        for step in range(n_trajectories.shape[1]):
            writer.writerow(
                [
                    step,
                    float(n_mean[step]),
                    float(n_std[step]),
                    float(s_mean[step]),
                    float(s_std[step]),
                    float(ratio_mean[step]),
                    float(ratio_std[step]),
                    float(ratio_q25[step]),
                    float(ratio_q50[step]),
                    float(ratio_q75[step]),
                ]
            )


def save_config(config: ExperimentConfig, *, output_dir: Path) -> None:
    (output_dir / "config.json").write_text(
        json.dumps(asdict(config), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def plot_s_mean_std(
    *,
    s_trajectories: np.ndarray,
    output_dir: Path,
) -> None:
    x = np.arange(s_trajectories.shape[1])
    s_mean = s_trajectories.mean(axis=0)
    s_std = s_trajectories.std(axis=0, ddof=1)

    fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
    ax.plot(x, s_mean, color="tab:green", linewidth=1.8, label="Mean of #S_k")
    ax.fill_between(
        x,
        s_mean - s_std,
        s_mean + s_std,
        color="tab:green",
        alpha=0.18,
        label=r"$\mathrm{mean} \pm \mathrm{std}$",
    )
    ax.set_xlabel("Шаг эволюции")
    ax.set_ylabel("#S_k")
    ax.set_title(r"Динамика числа информированных узлов")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.savefig(output_dir / "sk_mean_std.png", dpi=180)
    plt.close(fig)


def plot_ratio_mean_std(
    *,
    ratios: np.ndarray,
    output_dir: Path,
) -> None:
    x = np.arange(ratios.shape[1])
    ratio_mean = ratios.mean(axis=0)
    ratio_std = ratios.std(axis=0, ddof=1)

    fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
    ax.plot(
        x,
        ratio_mean,
        color="tab:blue",
        linewidth=1.8,
        label="Mean of #S_k / N_k",
    )
    ax.fill_between(
        x,
        np.clip(ratio_mean - ratio_std, 0.0, 1.0),
        np.clip(ratio_mean + ratio_std, 0.0, 1.0),
        color="tab:blue",
        alpha=0.18,
        label=r"$\mathrm{mean} \pm \mathrm{std}$",
    )
    ax.set_xlabel("Шаг эволюции")
    ax.set_ylabel("#S_k / N_k")
    ax.set_title(r"Динамика доли информированных узлов")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.savefig(output_dir / "sk_ratio_mean_std.png", dpi=180)
    plt.close(fig)


def plot_ratio_boxplots(
    *,
    ratios: np.ndarray,
    config: ExperimentConfig,
    output_dir: Path,
) -> None:
    steps = [step for step in config.boxplot_steps if 0 <= step <= config.steps]
    if not steps:
        return

    fig, ax = plt.subplots(figsize=(7.5, 4.5), layout="constrained")
    ax.boxplot(
        [ratios[:, step] for step in steps],
        tick_labels=[str(step) for step in steps],
        widths=0.5,
        patch_artist=True,
        boxprops={"facecolor": "#cfe3ff", "edgecolor": "black"},
        medianprops={"color": "black"},
        whiskerprops={"color": "black"},
        capprops={"color": "black"},
    )
    ax.set_xlabel("Шаг эволюции")
    ax.set_ylabel("#S_k / N_k")
    ax.set_title(r"Разброс доли информированных узлов на выбранных шагах")
    ax.grid(alpha=0.25, axis="y")
    fig.savefig(output_dir / "sk_ratio_boxplots.png", dpi=180)
    plt.close(fig)


def run_experiment(config: ExperimentConfig) -> Path:
    output_dir = Path(config.output_dir) / (f"p_{config.p:g}_c_{config.c:g}_runs_{config.runs}_steps_{config.steps}")
    output_dir.mkdir(parents=True, exist_ok=True)

    n_trajectories, s_trajectories, ratios = simulate_trajectories(config)

    save_config(config, output_dir=output_dir)
    save_trajectories_csv(
        n_trajectories=n_trajectories,
        s_trajectories=s_trajectories,
        ratios=ratios,
        output_dir=output_dir,
    )
    save_summary_csv(
        n_trajectories=n_trajectories,
        s_trajectories=s_trajectories,
        ratios=ratios,
        output_dir=output_dir,
    )
    plot_s_mean_std(s_trajectories=s_trajectories, output_dir=output_dir)
    plot_ratio_mean_std(ratios=ratios, output_dir=output_dir)
    plot_ratio_boxplots(ratios=ratios, config=config, output_dir=output_dir)

    return output_dir


def main() -> None:
    config = parse_args()
    output_dir = run_experiment(config)
    print(f"Saved S_k experiment artifacts to: {output_dir}")


if __name__ == "__main__":
    main()
