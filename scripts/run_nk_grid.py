#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path

from scripts.run_nk_experiment import ExperimentConfig, run_experiment


def parse_float_list(raw: str) -> list[float]:
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def parse_int_list(raw: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in raw.split(",") if item.strip())


def read_summary_rows(summary_path: Path) -> list[dict[str, str]]:
    with summary_path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_combined_summary(
    *,
    batch_dir: Path,
    experiment_rows: list[tuple[ExperimentConfig, Path, list[dict[str, str]]]],
) -> None:
    output_path = batch_dir / "combined_nk_summary.csv"
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "p",
                "c",
                "runs",
                "steps",
                "step",
                "empirical_mean",
                "theoretical_mean",
                "empirical_std",
                "empirical_var",
                "theoretical_var",
                "mean_abs_error",
                "var_abs_error",
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
                        row["empirical_mean"],
                        row["theoretical_mean"],
                        row["empirical_std"],
                        row["empirical_var"],
                        row["theoretical_var"],
                        row["mean_abs_error"],
                        row["var_abs_error"],
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
                "final_empirical_mean",
                "final_theoretical_mean",
                "final_empirical_var",
                "final_theoretical_var",
                "final_mean_abs_error",
                "final_var_abs_error",
                "max_mean_abs_error_over_steps",
                "max_var_abs_error_over_steps",
                "experiment_dir",
            ]
        )

        for config, experiment_dir, rows in experiment_rows:
            final_row = rows[-1]
            max_mean_abs_error = max(float(row["mean_abs_error"]) for row in rows)
            max_var_abs_error = max(float(row["var_abs_error"]) for row in rows)
            writer.writerow(
                [
                    config.p,
                    config.c,
                    config.runs,
                    config.steps,
                    final_row["empirical_mean"],
                    final_row["theoretical_mean"],
                    final_row["empirical_var"],
                    final_row["theoretical_var"],
                    final_row["mean_abs_error"],
                    final_row["var_abs_error"],
                    max_mean_abs_error,
                    max_var_abs_error,
                    str(experiment_dir),
                ]
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=("Run a grid of N_k Monte Carlo experiments and aggregate their summaries.")
    )
    parser.add_argument("--p-values", default="0.1,0.5,0.9")
    parser.add_argument("--c-values", default="0.1,0.2,0.3")
    parser.add_argument("--n0", type=int, default=2)
    parser.add_argument("--runs", type=int, default=300)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--delta-in", type=int, default=1)
    parser.add_argument("--delta-out", type=int, default=1)
    parser.add_argument("--truncation-k", type=int, default=10_000)
    parser.add_argument("--output-root", default="results/nk/grid")
    parser.add_argument(
        "--hist-steps",
        default="20,100,200",
        help="Comma-separated list of k values for histogram/pmf comparisons.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    p_values = parse_float_list(args.p_values)
    c_values = parse_float_list(args.c_values)
    hist_steps = parse_int_list(args.hist_steps)

    batch_dir = Path(args.output_root) / (
        f"grid_runs_{args.runs}_steps_{args.steps}"
        f"_p_{'-'.join(f'{value:g}' for value in p_values)}"
        f"_c_{'-'.join(f'{value:g}' for value in c_values)}"
    )
    batch_dir.mkdir(parents=True, exist_ok=True)

    batch_config = {
        "p_values": p_values,
        "c_values": c_values,
        "n0": args.n0,
        "runs": args.runs,
        "steps": args.steps,
        "seed": args.seed,
        "delta_in": args.delta_in,
        "delta_out": args.delta_out,
        "truncation_k": args.truncation_k,
        "hist_steps": hist_steps,
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
                n0=args.n0,
                runs=args.runs,
                steps=args.steps,
                seed=args.seed + 10_000 * p_index + 100 * c_index,
                delta_in=args.delta_in,
                delta_out=args.delta_out,
                truncation_k=args.truncation_k,
                output_dir=str(batch_dir),
                hist_steps=hist_steps,
            )
            experiment_dir = run_experiment(config)
            summary_rows = read_summary_rows(experiment_dir / "nk_summary.csv")
            experiment_rows.append((config, experiment_dir, summary_rows))
            print(f"Completed experiment p={config.p:g}, c={config.c:g} -> {experiment_dir}")

    write_combined_summary(batch_dir=batch_dir, experiment_rows=experiment_rows)
    write_final_step_summary(batch_dir=batch_dir, experiment_rows=experiment_rows)

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
