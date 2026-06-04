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

from main import make_default_graph, sample_kstar_over_runs

matplotlib.use("Agg")


@dataclass(slots=True)
class ExperimentConfig:
    p: float = 0.5
    c: float = 0.2
    runs: int = 200
    seed: int = 42
    delta_in: int = 1
    delta_out: int = 1
    truncation_k: int = 10_000
    output_dir: str = "results/sk_kstar"
    nu: float = 1.0
    t_star: float = 200.0
    theta_values: tuple[float, ...] = (0.7, 0.8, 0.9, 0.95)


def parse_float_list(raw: str) -> tuple[float, ...]:
    return tuple(float(item.strip()) for item in raw.split(",") if item.strip())


def parse_args() -> ExperimentConfig:
    parser = argparse.ArgumentParser(
        description=("Run a Monte Carlo experiment for |S_{K*}| and |S_{K*}| / N_{K*} when K* ~ Poisson(nu * T*).")
    )
    parser.add_argument("--p", type=float, default=0.5)
    parser.add_argument("--c", type=float, default=0.2)
    parser.add_argument("--runs", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--delta-in", type=int, default=1)
    parser.add_argument("--delta-out", type=int, default=1)
    parser.add_argument("--truncation-k", type=int, default=10_000)
    parser.add_argument("--output-dir", default="results/sk_kstar")
    parser.add_argument("--nu", type=float, default=1.0)
    parser.add_argument("--t-star", type=float, default=200.0)
    parser.add_argument("--theta-values", default="0.7,0.8,0.9,0.95")
    args = parser.parse_args()

    return ExperimentConfig(
        p=args.p,
        c=args.c,
        runs=args.runs,
        seed=args.seed,
        delta_in=args.delta_in,
        delta_out=args.delta_out,
        truncation_k=args.truncation_k,
        output_dir=args.output_dir,
        nu=args.nu,
        t_star=args.t_star,
        theta_values=parse_float_list(args.theta_values),
    )


def sample_std(values: np.ndarray) -> float:
    if values.size <= 1:
        return 0.0
    return float(values.std(ddof=1))


def poisson_mean(config: ExperimentConfig) -> float:
    return config.nu * config.t_star


def simulate_final_samples(
    config: ExperimentConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    step_rng = np.random.default_rng(config.seed)
    k_star = step_rng.poisson(lam=poisson_mean(config), size=config.runs).astype(np.int64)
    n_kstar, s_kstar, ratio_kstar = sample_kstar_over_runs(
        make_graph=lambda run_idx: make_default_graph(
            c=config.c,
            p=config.p,
            delta_in=config.delta_in,
            delta_out=config.delta_out,
            rng=np.random.default_rng(config.seed + 1_000_000 + run_idx),
        ),
        k_star=k_star,
        truncation_k=config.truncation_k,
    )
    return k_star, n_kstar, s_kstar, ratio_kstar


def save_config(config: ExperimentConfig, *, output_dir: Path) -> None:
    (output_dir / "config.json").write_text(
        json.dumps(asdict(config), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def save_samples_csv(
    *,
    k_star: np.ndarray,
    n_kstar: np.ndarray,
    s_kstar: np.ndarray,
    ratio_kstar: np.ndarray,
    output_dir: Path,
) -> None:
    output_path = output_dir / "kstar_samples.csv"
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["run_id", "K_star", "N_K_star", "S_K_star", "ratio_K_star"])
        for run_id in range(k_star.size):
            writer.writerow(
                [
                    run_id + 1,
                    int(k_star[run_id]),
                    int(n_kstar[run_id]),
                    int(s_kstar[run_id]),
                    float(ratio_kstar[run_id]),
                ]
            )


def save_summary_csv(
    *,
    config: ExperimentConfig,
    k_star: np.ndarray,
    n_kstar: np.ndarray,
    s_kstar: np.ndarray,
    ratio_kstar: np.ndarray,
    output_dir: Path,
) -> None:
    output_path = output_dir / "kstar_summary.csv"
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
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
            ]
        )
        writer.writerow(
            [
                poisson_mean(config),
                int(k_star.max(initial=0)),
                float(k_star.mean()),
                sample_std(k_star.astype(np.float64)),
                float(np.quantile(k_star, 0.25)),
                float(np.quantile(k_star, 0.50)),
                float(np.quantile(k_star, 0.75)),
                float(n_kstar.mean()),
                sample_std(n_kstar.astype(np.float64)),
                float(s_kstar.mean()),
                sample_std(s_kstar.astype(np.float64)),
                float(ratio_kstar.mean()),
                sample_std(ratio_kstar),
                float(np.quantile(ratio_kstar, 0.05)),
                float(np.quantile(ratio_kstar, 0.25)),
                float(np.quantile(ratio_kstar, 0.50)),
                float(np.quantile(ratio_kstar, 0.75)),
                float(np.quantile(ratio_kstar, 0.95)),
            ]
        )


def save_thresholds_csv(
    *,
    config: ExperimentConfig,
    ratio_kstar: np.ndarray,
    output_dir: Path,
) -> None:
    output_path = output_dir / "coverage_thresholds.csv"
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["theta", "count_ge_theta", "prob_ge_theta"])
        for theta in config.theta_values:
            count = int(np.count_nonzero(ratio_kstar >= theta))
            probability = count / ratio_kstar.size if ratio_kstar.size else 0.0
            writer.writerow([theta, count, probability])


def plot_ratio_histogram(*, ratio_kstar: np.ndarray, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.5), layout="constrained")
    ax.hist(
        ratio_kstar,
        bins=np.linspace(0.0, 1.0, 26),
        color="#7fb3d5",
        edgecolor="white",
        alpha=0.85,
    )
    ax.set_xlabel(r"$|S_{K^{*}}| / N_{K^{*}}$")
    ax.set_ylabel("Число траекторий")
    ax.set_title(r"Эмпирическое распределение доли информированных узлов к моменту $T^{*}$")
    ax.grid(alpha=0.25, axis="y")
    fig.savefig(output_dir / "ratio_kstar_histogram.png", dpi=180)
    plt.close(fig)


def plot_ratio_ecdf(*, ratio_kstar: np.ndarray, output_dir: Path) -> None:
    sorted_values = np.sort(ratio_kstar)
    probabilities = np.arange(1, sorted_values.size + 1, dtype=np.float64) / sorted_values.size

    fig, ax = plt.subplots(figsize=(7.5, 4.5), layout="constrained")
    ax.step(sorted_values, probabilities, where="post", color="#1f618d", linewidth=1.8)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$P(|S_{K^{*}}| / N_{K^{*}} \leq x)$")
    ax.set_title(r"Эмпирическая функция распределения для $|S_{K^{*}}| / N_{K^{*}}$")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.25)
    fig.savefig(output_dir / "ratio_kstar_ecdf.png", dpi=180)
    plt.close(fig)


def plot_s_histogram(*, s_kstar: np.ndarray, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.5), layout="constrained")
    bins = min(25, max(10, int(np.sqrt(max(s_kstar.size, 1)))))
    ax.hist(
        s_kstar,
        bins=bins,
        color="#73c6b6",
        edgecolor="white",
        alpha=0.85,
    )
    ax.set_xlabel(r"$|S_{K^{*}}|$")
    ax.set_ylabel("Число траекторий")
    ax.set_title(r"Эмпирическое распределение числа информированных узлов к моменту $T^{*}$")
    ax.grid(alpha=0.25, axis="y")
    fig.savefig(output_dir / "s_kstar_histogram.png", dpi=180)
    plt.close(fig)


def run_experiment(config: ExperimentConfig) -> Path:
    output_dir = Path(config.output_dir) / (
        f"p_{config.p:g}_c_{config.c:g}_runs_{config.runs}_nu_{config.nu:g}_t_{config.t_star:g}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    k_star, n_kstar, s_kstar, ratio_kstar = simulate_final_samples(config)

    save_config(config, output_dir=output_dir)
    save_samples_csv(
        k_star=k_star,
        n_kstar=n_kstar,
        s_kstar=s_kstar,
        ratio_kstar=ratio_kstar,
        output_dir=output_dir,
    )
    save_summary_csv(
        config=config,
        k_star=k_star,
        n_kstar=n_kstar,
        s_kstar=s_kstar,
        ratio_kstar=ratio_kstar,
        output_dir=output_dir,
    )
    save_thresholds_csv(config=config, ratio_kstar=ratio_kstar, output_dir=output_dir)
    plot_ratio_histogram(ratio_kstar=ratio_kstar, output_dir=output_dir)
    plot_ratio_ecdf(ratio_kstar=ratio_kstar, output_dir=output_dir)
    plot_s_histogram(s_kstar=s_kstar, output_dir=output_dir)

    return output_dir


def main() -> None:
    config = parse_args()
    output_dir = run_experiment(config)
    print(f"Saved K* S_k experiment artifacts to: {output_dir}")


if __name__ == "__main__":
    main()
