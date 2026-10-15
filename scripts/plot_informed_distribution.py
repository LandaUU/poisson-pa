from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results/informed_distribution_input"
OUT = ROOT / "results/informed_distribution_output"
FIGURE = ROOT / "Fig17.png"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    OUT.mkdir(exist_ok=True)
    reference = json.loads((SOURCE / "reference.json").read_text())
    with np.load(SOURCE / "samples.npz") as archive:
        samples = archive["samples"]
    assert samples.shape == (18086, 1, 5)
    values = samples[:, 0, 2].astype(int)
    assert np.all(values == samples[:, 0, 2])

    mass = defaultdict(float)
    for cell in reference["unnormalized_joint"]:
        mass[cell["state"][2]] += cell["mass"]
    assert abs(sum(mass.values()) - 1) < 1e-12

    labels = ["1", "2", "3", "4", "5", r"$\geq 6$"]
    theory = [mass[k] for k in range(1, 6)] + [sum(v for k, v in mass.items() if k >= 6)]
    counts = [int(np.count_nonzero(values == k)) for k in range(1, 6)]
    counts.append(int(np.count_nonzero(values >= 6)))
    n = len(values)
    empirical = np.array(counts) / n
    # Per-category Wilson intervals; the six intervals are pointwise, not simultaneous.
    intervals = np.array([binomtest(c, n).proportion_ci(0.95, method="wilson") for c in counts])

    with (OUT / "frequencies.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "S1",
                "exact_probability",
                "count",
                "empirical_probability",
                "wilson_95_lower",
                "wilson_95_upper",
            ]
        )
        for label, expected, count, observed, ci in zip(labels, theory, counts, empirical, intervals):
            writer.writerow(
                [
                    label.replace("$", ""),
                    f"{expected:.17g}",
                    count,
                    f"{observed:.17g}",
                    f"{ci[0]:.17g}",
                    f"{ci[1]:.17g}",
                ]
            )

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(7.4, 4.1), constrained_layout=True)
    ax.bar(x - 0.18, theory, width=0.34, color="#225ea8", label="Exact finite-state law")
    ax.bar(x + 0.18, empirical, width=0.34, color="#fd8d3c", label="Simulation")
    ax.errorbar(
        x + 0.18,
        empirical,
        yerr=[empirical - intervals[:, 0], intervals[:, 1] - empirical],
        fmt="none",
        ecolor="#4d4d4d",
        capsize=3,
        linewidth=1,
        label="Pointwise 95% Wilson CI",
    )
    ax.set_xticks(x, labels)
    ax.set_xlabel(r"Number of informed nodes $S_1$")
    ax.set_ylabel("Probability")
    ax.set_ylim(0, 0.59)
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.2)
    fig.savefig(FIGURE, dpi=220)
    plt.close(fig)

    summary = {
        "source_case": SOURCE.name,
        "initial_graph": {
            "vertices": [0, 1],
            "edges": [[0, 0], [1, 1]],
            "informed": [0],
        },
        "p": 0.5,
        "lambda_1": 0.5,
        "delta_in": 1,
        "delta_out": 1,
        "step": 1,
        "replicates": n,
        "max_absolute_category_difference": float(np.max(np.abs(empirical - theory))),
        "tail_probability_Poisson_ge_16": 4.55718e-19,
        "intervals": "pointwise 95% Wilson for independent trajectory frequencies",
        "source_sha256": {name: sha256(SOURCE / name) for name in ("reference.json", "samples.npz")},
        "output_sha256": {
            "frequencies.csv": sha256(OUT / "frequencies.csv"),
            "Fig17.png": sha256(FIGURE),
        },
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(
        json.dumps(
            {
                "replicates": n,
                "max_difference": summary["max_absolute_category_difference"],
                "figure": str(FIGURE),
            }
        )
    )


if __name__ == "__main__":
    main()
