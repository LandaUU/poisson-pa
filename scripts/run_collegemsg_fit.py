#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-master-deg-project")

import matplotlib
import numpy as np

matplotlib.use("Agg")

from model_fitting import (  # noqa: E402
    MessageRule,
    angular_bandwidth_diagnostic,
    angular_samples,
    degree_arrays_from_edges,
    filter_edges,
    fit_poisson_pa_parameters,
    fit_to_json_dict,
    localized_plot_path,
    plot_angular_density_comparison,
    plot_degree_tail_comparison,
    plot_empirical_message_dynamics,
    plot_message_coverage_comparison,
    read_temporal_edges,
    simulate_fitted_model,
    temporal_message_coverage,
    truncation_for_lambda,
    write_fit_summary,
    write_real_coverage_csv,
    write_simulated_trajectories_csv,
)


@dataclass(slots=True)
class CollegeMsgFitConfig:
    data_path: str
    output_dir: str = "results/collegemsg_fit"
    dataset_label: str = "observed"
    bucket_seconds: int = 3600
    hour_mode: str = "active"
    min_tail: int = 10
    max_tail_fraction: float = 0.5
    angular_quantile: float = 0.995
    real_message_rule: MessageRule = "target_to_source"
    synthetic_message_rule: MessageRule = "target_to_source"
    runs: int = 20
    seed: int = 42
    truncation_k: int | None = None
    start_timestamp: int | None = None
    end_timestamp: int | None = None
    skip_simulation: bool = False


def parse_args() -> CollegeMsgFitConfig:
    parser = argparse.ArgumentParser(
        description=(
            "Fit the Wang-Resnick Poisson preferential attachment model to a SNAP-style temporal dataset "
            "using the extreme-value approach from the model fitting section."
        )
    )
    parser.add_argument("--data-path", required=True, help="Path to a SNAP-style temporal edge list, plain text or .gz")
    parser.add_argument("--output-dir", default="results/collegemsg_fit")
    parser.add_argument("--dataset-label", default="observed", help="Label used for the empirical dataset in plots.")
    parser.add_argument("--bucket-seconds", type=int, default=3600)
    parser.add_argument("--hour-mode", choices=("active", "span"), default="active")
    parser.add_argument("--min-tail", type=int, default=10)
    parser.add_argument("--max-tail-fraction", type=float, default=0.5)
    parser.add_argument("--angular-quantile", type=float, default=0.995)
    parser.add_argument(
        "--real-message-rule",
        choices=("source_to_target", "target_to_source"),
        default="target_to_source",
        help="Rule used for the real temporal coverage curve.",
    )
    parser.add_argument(
        "--synthetic-message-rule",
        choices=("source_to_target", "target_to_source"),
        default="target_to_source",
        help="Rule used for message propagation on synthetic fitted trajectories.",
    )
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--truncation-k", type=int, default=None)
    parser.add_argument("--start-timestamp", type=int, default=None)
    parser.add_argument("--end-timestamp", type=int, default=None)
    parser.add_argument("--skip-simulation", action="store_true")
    args = parser.parse_args()

    return CollegeMsgFitConfig(
        data_path=args.data_path,
        output_dir=args.output_dir,
        dataset_label=args.dataset_label,
        bucket_seconds=args.bucket_seconds,
        hour_mode=args.hour_mode,
        min_tail=args.min_tail,
        max_tail_fraction=args.max_tail_fraction,
        angular_quantile=args.angular_quantile,
        real_message_rule=args.real_message_rule,
        synthetic_message_rule=args.synthetic_message_rule,
        runs=args.runs,
        seed=args.seed,
        truncation_k=args.truncation_k,
        start_timestamp=args.start_timestamp,
        end_timestamp=args.end_timestamp,
        skip_simulation=args.skip_simulation,
    )


def run(config: CollegeMsgFitConfig) -> Path:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    edges = filter_edges(
        read_temporal_edges(config.data_path),
        start_timestamp=config.start_timestamp,
        end_timestamp=config.end_timestamp,
    )
    real_in, real_out, _nodes = degree_arrays_from_edges(edges)
    fit = fit_poisson_pa_parameters(
        edges,
        bucket_seconds=config.bucket_seconds,
        hour_mode=config.hour_mode,  # type: ignore[arg-type]
        min_tail=config.min_tail,
        max_tail_fraction=config.max_tail_fraction,
    )

    coverages = [
        temporal_message_coverage(edges, rule="source_to_target", bucket_seconds=config.bucket_seconds),
        temporal_message_coverage(edges, rule="target_to_source", bucket_seconds=config.bucket_seconds),
    ]
    coverage_by_rule = {coverage.rule: coverage for coverage in coverages}
    selected_coverage = coverage_by_rule[config.real_message_rule]

    simulated_runs = []
    skipped_reason = None
    if config.skip_simulation:
        skipped_reason = "disabled by --skip-simulation"
    elif fit.delta_in_hat <= 0.0 or fit.delta_out_hat <= 0.0:
        skipped_reason = f"non-positive fitted delta: delta_in={fit.delta_in_hat:g}, delta_out={fit.delta_out_hat:g}"
    else:
        simulated_runs = simulate_fitted_model(
            fit,
            runs=config.runs,
            seed=config.seed,
            truncation_k=config.truncation_k or truncation_for_lambda(fit.lambda_hat),
            message_rule=config.synthetic_message_rule,
        )

    a_hat = fit.iota_in_hat / fit.iota_out_hat
    real_angles = angular_samples(real_in, real_out, a=a_hat, quantile=config.angular_quantile)
    simulated_angles = [
        angular_samples(run.in_degrees, run.out_degrees, a=a_hat, quantile=config.angular_quantile)
        for run in simulated_runs
    ]
    nonempty_simulated_angles = [angles for angles in simulated_angles if angles.size]
    pooled_simulated_angles = np.concatenate(nonempty_simulated_angles) if nonempty_simulated_angles else np.empty(0)
    angular_bandwidth = {
        "angular_quantile": float(config.angular_quantile),
        "a_hat": float(a_hat),
        "diagnostics": [
            angular_bandwidth_diagnostic(real_angles, label="real", seed=config.seed + 10_000),
            angular_bandwidth_diagnostic(pooled_simulated_angles, label="simulated_pooled", seed=config.seed + 10_001),
        ],
    }

    write_real_coverage_csv(coverages, output_dir / "real_message_coverage.csv")
    write_simulated_trajectories_csv(simulated_runs, output_dir / "simulated_message_trajectories.csv")
    (output_dir / "angular_bandwidth_diagnostic.json").write_text(
        json.dumps(angular_bandwidth, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    empirical_dynamics_coverage = temporal_message_coverage(
        edges,
        rule=config.real_message_rule,
        bucket_seconds=24 * 3600,
    )
    for language in ("en", "ru"):
        degree_output = output_dir / "degree_tail_compare.png"
        angular_output = output_dir / "angular_density_compare.png"
        coverage_output = output_dir / "message_coverage_compare.png"
        dynamics_output = output_dir / "empirical_dynamics.png"

        plot_degree_tail_comparison(
            real_in_degrees=real_in,
            real_out_degrees=real_out,
            simulated_runs=simulated_runs,
            output_path=localized_plot_path(degree_output, language),
            dataset_label=config.dataset_label,
            language=language,
        )
        plot_angular_density_comparison(
            real_angles=real_angles,
            simulated_angles=simulated_angles,
            output_path=localized_plot_path(angular_output, language),
            dataset_label=config.dataset_label,
            language=language,
        )
        plot_message_coverage_comparison(
            real_coverage=selected_coverage,
            simulated_runs=simulated_runs,
            output_path=localized_plot_path(coverage_output, language),
            language=language,
        )
        plot_empirical_message_dynamics(
            real_coverage=empirical_dynamics_coverage,
            output_path=localized_plot_path(dynamics_output, language),
            language=language,
        )

    plot_degree_tail_comparison(
        real_in_degrees=real_in,
        real_out_degrees=real_out,
        simulated_runs=simulated_runs,
        output_path=output_dir / "degree_tail_compare.png",
        dataset_label=config.dataset_label,
        language="en",
    )
    plot_angular_density_comparison(
        real_angles=real_angles,
        simulated_angles=simulated_angles,
        output_path=output_dir / "angular_density_compare.png",
        dataset_label=config.dataset_label,
        language="en",
    )
    plot_message_coverage_comparison(
        real_coverage=selected_coverage,
        simulated_runs=simulated_runs,
        output_path=output_dir / "message_coverage_compare.png",
        language="en",
    )
    plot_empirical_message_dynamics(
        real_coverage=empirical_dynamics_coverage,
        output_path=output_dir / "empirical_dynamics.png",
        language="en",
    )

    summary = fit_to_json_dict(
        fit=fit,
        selected_coverage=selected_coverage,
        all_coverages=coverages,
        simulated_runs=simulated_runs,
        config=asdict(config),
        skipped_simulation_reason=skipped_reason,
    )
    write_fit_summary(summary, output_dir / "fit_summary.json")
    return output_dir


def main() -> None:
    config = parse_args()
    output_dir = run(config)
    print(f"Saved fitting artifacts to: {output_dir}")


if __name__ == "__main__":
    main()
