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
from scipy.stats import binom, poisson

from main import Graph, lambda_f

matplotlib.use("Agg")


@dataclass(slots=True)
class ExperimentConfig:
    p: float = 0.5
    c: float = 0.1
    n0: int = 2
    runs: int = 100
    steps: int = 200
    seed: int = 42
    delta_in: int = 1
    delta_out: int = 1
    truncation_k: int = 10_000
    output_dir: str = "results/nk"
    hist_steps: tuple[int, ...] = (20, 100, 200)


def make_sampler(config: ExperimentConfig, *, seed: int) -> Graph:
    return Graph(
        c=config.c,
        p=config.p,
        delta_in=config.delta_in,
        delta_out=config.delta_out,
        rng=np.random.default_rng(seed),
    )


def simulate_nk_trajectories(config: ExperimentConfig) -> np.ndarray:
    trajectories = np.empty((config.runs, config.steps + 1), dtype=np.int64)
    trajectories[:, 0] = config.n0

    for run_idx in range(config.runs):
        sampler = make_sampler(config, seed=config.seed + run_idx)
        n_current = config.n0

        for step in range(1, config.steps + 1):
            delta_m = sampler.sample_delta_m(
                current_round=step,
                k=config.truncation_k,
            )
            delta_n = int(sampler.rng.binomial(delta_m, sampler.params.p))
            n_current += delta_n
            trajectories[run_idx, step] = n_current

    return trajectories


def theoretical_moments(
    config: ExperimentConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    steps = np.arange(config.steps + 1, dtype=np.int64)
    lambda_values = np.array(
        [0.0] + [lambda_f(step, config.c) for step in range(1, config.steps + 1)],
        dtype=np.float64,
    )
    lambda_prefix = np.cumsum(lambda_values)
    mean = config.n0 + steps * config.p + config.p * lambda_prefix
    variance = steps * config.p * (1.0 - config.p) + config.p * lambda_prefix
    return lambda_prefix, mean, variance


def empirical_summary(
    trajectories: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    empirical_mean = trajectories.mean(axis=0)
    empirical_std = trajectories.std(axis=0, ddof=1)
    empirical_var = trajectories.var(axis=0, ddof=1)
    return empirical_mean, empirical_std, empirical_var


def theoretical_nk_pmf(
    n_values: np.ndarray,
    *,
    step: int,
    config: ExperimentConfig,
    lambda_prefix: np.ndarray,
) -> np.ndarray:
    poisson_mean = config.p * lambda_prefix[step]
    pmf = np.zeros_like(n_values, dtype=np.float64)

    for idx, n in enumerate(n_values):
        offset = int(n - config.n0)
        if offset < 0:
            continue

        b = np.arange(min(step, offset) + 1, dtype=np.int64)
        pmf[idx] = np.sum(binom.pmf(b, step, config.p) * poisson.pmf(offset - b, poisson_mean))

    return pmf


def save_trajectories_csv(trajectories: np.ndarray, *, output_dir: Path) -> None:
    csv_path = output_dir / "nk_trajectories.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["run_id", "step", "N_k"])
        for run_id, trajectory in enumerate(trajectories, start=1):
            for step, n_value in enumerate(trajectory):
                writer.writerow([run_id, step, int(n_value)])


def save_summary_csv(
    *,
    output_dir: Path,
    empirical_mean: np.ndarray,
    empirical_std: np.ndarray,
    empirical_var: np.ndarray,
    theoretical_mean: np.ndarray,
    theoretical_var: np.ndarray,
) -> None:
    csv_path = output_dir / "nk_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "step",
                "empirical_mean",
                "theoretical_mean",
                "empirical_std",
                "empirical_var",
                "theoretical_var",
                "mean_abs_error",
                "var_abs_error",
            ]
        )

        for step in range(len(empirical_mean)):
            writer.writerow(
                [
                    step,
                    float(empirical_mean[step]),
                    float(theoretical_mean[step]),
                    float(empirical_std[step]),
                    float(empirical_var[step]),
                    float(theoretical_var[step]),
                    float(abs(empirical_mean[step] - theoretical_mean[step])),
                    float(abs(empirical_var[step] - theoretical_var[step])),
                ]
            )


def save_config(config: ExperimentConfig, *, output_dir: Path) -> None:
    config_path = output_dir / "config.json"
    config_path.write_text(
        json.dumps(asdict(config), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def plot_mean_comparison(
    *,
    output_dir: Path,
    empirical_mean: np.ndarray,
    empirical_std: np.ndarray,
    theoretical_mean: np.ndarray,
) -> None:
    x = np.arange(len(empirical_mean))
    fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
    ax.plot(x, theoretical_mean, label="Theory E(N_k)", color="black", linewidth=1.8)
    ax.plot(x, empirical_mean, label="Empirical mean", color="tab:blue", linewidth=1.5)
    ax.fill_between(
        x,
        empirical_mean - empirical_std,
        empirical_mean + empirical_std,
        color="tab:blue",
        alpha=0.18,
        label="Empirical mean ± std",
    )
    ax.set_xlabel("k")
    ax.set_ylabel("N_k")
    ax.set_title("N_k: empirical mean vs theoretical expectation")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.savefig(output_dir / "nk_mean_vs_theory.png", dpi=180)
    plt.close(fig)


def plot_variance_comparison(
    *,
    output_dir: Path,
    empirical_var: np.ndarray,
    theoretical_var: np.ndarray,
) -> None:
    x = np.arange(len(empirical_var))
    fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
    ax.plot(x, theoretical_var, label="Theory Var(N_k)", color="black", linewidth=1.8)
    ax.plot(
        x,
        empirical_var,
        label="Empirical variance",
        color="tab:orange",
        linewidth=1.5,
    )
    ax.set_xlabel("k")
    ax.set_ylabel("Var(N_k)")
    ax.set_title("N_k: empirical variance vs theoretical variance")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.savefig(output_dir / "nk_variance_vs_theory.png", dpi=180)
    plt.close(fig)


def plot_histograms(
    *,
    output_dir: Path,
    trajectories: np.ndarray,
    config: ExperimentConfig,
    lambda_prefix: np.ndarray,
) -> None:
    valid_hist_steps = [step for step in config.hist_steps if 0 <= step <= config.steps]

    for step in valid_hist_steps:
        samples = trajectories[:, step]
        x_min = int(samples.min())
        x_max = int(samples.max())
        x_values = np.arange(x_min, x_max + 1, dtype=np.int64)
        pmf = theoretical_nk_pmf(
            x_values,
            step=step,
            config=config,
            lambda_prefix=lambda_prefix,
        )

        fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
        bins = np.arange(x_min - 0.5, x_max + 1.5, 1.0)
        weights = np.full(samples.shape[0], 1.0 / samples.shape[0], dtype=np.float64)
        ax.hist(
            samples,
            bins=bins,
            weights=weights,
            alpha=0.55,
            color="tab:blue",
            edgecolor="white",
            label="Empirical histogram",
        )
        ax.plot(
            x_values,
            pmf,
            marker="o",
            markersize=3.5,
            linewidth=1.2,
            color="black",
            label="Theory pmf",
        )
        ax.set_xlabel("N_k")
        ax.set_ylabel("Probability")
        ax.set_title(f"N_k distribution at k = {step}")
        ax.grid(alpha=0.25)
        ax.legend()
        fig.savefig(output_dir / f"nk_hist_step_{step}.png", dpi=180)
        plt.close(fig)


def parse_args() -> ExperimentConfig:
    parser = argparse.ArgumentParser(
        description=("Run a Monte Carlo experiment for N_k using the same sampling functions as main.py.")
    )
    parser.add_argument("--p", type=float, default=0.5)
    parser.add_argument("--c", type=float, default=0.1)
    parser.add_argument("--n0", type=int, default=2)
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--delta-in", type=int, default=1)
    parser.add_argument("--delta-out", type=int, default=1)
    parser.add_argument("--truncation-k", type=int, default=10_000)
    parser.add_argument("--output-dir", default="results/nk")
    parser.add_argument(
        "--hist-steps",
        default="20,100,200",
        help="Comma-separated list of k values for histogram/pmf comparisons.",
    )
    args = parser.parse_args()

    hist_steps = tuple(int(item.strip()) for item in args.hist_steps.split(",") if item.strip())
    return ExperimentConfig(
        p=args.p,
        c=args.c,
        n0=args.n0,
        runs=args.runs,
        steps=args.steps,
        seed=args.seed,
        delta_in=args.delta_in,
        delta_out=args.delta_out,
        truncation_k=args.truncation_k,
        output_dir=args.output_dir,
        hist_steps=hist_steps,
    )


def run_experiment(config: ExperimentConfig) -> Path:
    output_dir = Path(config.output_dir) / (f"p_{config.p:g}_c_{config.c:g}_runs_{config.runs}_steps_{config.steps}")
    output_dir.mkdir(parents=True, exist_ok=True)

    trajectories = simulate_nk_trajectories(config)
    lambda_prefix, theoretical_mean, theoretical_var = theoretical_moments(config)
    empirical_mean, empirical_std, empirical_var = empirical_summary(trajectories)

    save_config(config, output_dir=output_dir)
    save_trajectories_csv(trajectories, output_dir=output_dir)
    save_summary_csv(
        output_dir=output_dir,
        empirical_mean=empirical_mean,
        empirical_std=empirical_std,
        empirical_var=empirical_var,
        theoretical_mean=theoretical_mean,
        theoretical_var=theoretical_var,
    )
    plot_mean_comparison(
        output_dir=output_dir,
        empirical_mean=empirical_mean,
        empirical_std=empirical_std,
        theoretical_mean=theoretical_mean,
    )
    plot_variance_comparison(
        output_dir=output_dir,
        empirical_var=empirical_var,
        theoretical_var=theoretical_var,
    )
    plot_histograms(
        output_dir=output_dir,
        trajectories=trajectories,
        config=config,
        lambda_prefix=lambda_prefix,
    )

    return output_dir


def main() -> None:
    config = parse_args()
    output_dir = run_experiment(config)
    print(f"Saved N_k experiment artifacts to: {output_dir}")


if __name__ == "__main__":
    main()
