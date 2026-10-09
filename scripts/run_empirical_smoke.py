"""Generate small current-model ensembles and render intervals without external archives."""

import argparse
import hashlib
import json
from pathlib import Path

from scripts.render_supervisor_revision import main as render
from scripts.run_nk_experiment import ExperimentConfig as NodeConfig
from scripts.run_nk_experiment import run_experiment as run_nodes
from scripts.run_pn_experiment import ExperimentConfig as ProbabilityConfig
from scripts.run_pn_experiment import run_experiment as run_probability
from scripts.run_sk_experiment import ExperimentConfig as CoverageConfig
from scripts.run_sk_experiment import run_experiment as run_coverage


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True, help="New directory; never overwrites saved results")
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    inputs = args.output_dir / "inputs"
    common = dict(p=0.5, c=0.2, runs=8, steps=6, seed=42)
    run_nodes(NodeConfig(**common, hist_steps=(6,), output_dir=str(inputs / "nk")))
    run_coverage(CoverageConfig(**common, boxplot_steps=(6,), output_dir=str(inputs / "sk")))
    pn = run_probability(
        ProbabilityConfig(
            p_values=(0.5,),
            c_values=(0.2,),
            snapshot_steps=(1, 3, 6),
            base_runs=2,
            replications=100,
            seed=42,
            output_dir=str(inputs / "pn"),
        )
    )
    render(
        [
            "--output-dir",
            str(args.output_dir / "intervals"),
            "--nk-dir",
            str(inputs / "nk"),
            "--sk-dir",
            str(inputs / "sk"),
            "--pn-dir",
            str(pn),
            "--nk-runs",
            "8",
            "--sk-runs",
            "8",
            "--steps",
            "6",
            "--p-values",
            "0.5",
            "--c-values",
            "0.2",
            "--bootstrap-resamples",
            "100",
        ]
    )
    source_names = (
        "main.py",
        "propagation.py",
        "scripts/run_nk_experiment.py",
        "scripts/run_sk_experiment.py",
        "scripts/run_pn_experiment.py",
        "scripts/run_empirical_smoke.py",
    )
    root = Path(__file__).resolve().parents[1]
    report = dict(
        scope="current-model pipeline smoke; not manuscript numerical reproduction, GOF or predictive validation",
        seed=42,
        runs=8,
        steps=6,
        p=0.5,
        c=0.2,
        source_sha256={name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in source_names},
    )
    (args.output_dir / "smoke.json").write_text(json.dumps(report, indent=2) + "\n")
    print(args.output_dir)


if __name__ == "__main__":
    main()
