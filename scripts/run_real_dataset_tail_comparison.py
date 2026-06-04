#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-master-deg-project")

import matplotlib

matplotlib.use("Agg")

from model_fitting import (  # noqa: E402
    degree_arrays_from_edges,
    localized_plot_path,
    plot_real_dataset_degree_tail_overlay,
    plot_real_dataset_out_tail_threshold_overlay,
    read_temporal_edges,
)


@dataclass(slots=True, frozen=True)
class DatasetConfig:
    label: str
    data_path: str
    summary_path: str


DEFAULT_DATASETS = (
    DatasetConfig("CollegeMsg", "data/CollegeMsg.txt.gz", "results/collegemsg_fit/fit_summary.json"),
    DatasetConfig(
        "email-Eu-core temporal",
        "data/email-Eu-core-temporal.txt.gz",
        "results/email_eu_core_temporal_fit/fit_summary.json",
    ),
    DatasetConfig(
        "MathOverflow a2q", "data/sx-mathoverflow-a2q.txt.gz", "results/mathoverflow_a2q_fit/fit_summary.json"
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot empirical degree-tail overlays for real temporal datasets used in model fitting."
    )
    parser.add_argument(
        "--output-dir",
        default="results/real_dataset_tail_comparison",
        help="Directory where comparison plots will be saved.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    datasets = []
    out_tail_datasets = []
    for config in DEFAULT_DATASETS:
        edges = read_temporal_edges(config.data_path)
        in_degrees, out_degrees, _nodes = degree_arrays_from_edges(edges)
        datasets.append((config.label, in_degrees, out_degrees))
        summary = json.loads(Path(config.summary_path).read_text(encoding="utf-8"))
        out_tail_datasets.append(
            (
                config.label,
                out_degrees,
                float(summary["fit"]["out_tail"]["threshold_value"]),
                float(summary["fit"]["iota_out_hat"]),
            )
        )

    degree_output_path = output_dir / "real_dataset_degree_tail_overlay.png"
    out_tail_output_path = output_dir / "real_dataset_out_tail_threshold_overlay.png"
    for language in ("en", "ru"):
        plot_real_dataset_degree_tail_overlay(
            datasets=datasets,
            output_path=localized_plot_path(degree_output_path, language),
            language=language,
        )
        plot_real_dataset_out_tail_threshold_overlay(
            datasets=out_tail_datasets,
            output_path=localized_plot_path(out_tail_output_path, language),
            language=language,
        )
    plot_real_dataset_degree_tail_overlay(datasets=datasets, output_path=degree_output_path, language="en")
    plot_real_dataset_out_tail_threshold_overlay(
        datasets=out_tail_datasets, output_path=out_tail_output_path, language="en"
    )
    print(f"Saved comparison plots to: {output_dir}")


if __name__ == "__main__":
    main()
