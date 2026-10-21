"""One-step repeated-contact experiment."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from collections import Counter
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy.stats import binomtest, poisson

from main import Graph
from propagation import batch_recipients


def contact_metrics(edges, old_nodes, informed):
    """Count successful contacts and distinct pull recipients."""
    hits = Counter(u for u, v in edges if v in informed and u in old_nodes and u not in informed)
    pairs = Counter((u, v) for u, v in edges if v in informed and u in old_nodes and u not in informed)
    successful = sum(v in informed and u not in informed for u, v in edges)
    distinct = len(batch_recipients(edges, informed))
    result = {
        "successful_contacts": successful,
        "distinct_newly_informed": distinct,
        "old_successful_contacts": sum(hits.values()),
        "old_distinct_recipients": len(hits),
        "old_multihit_recipients": sum(n >= 2 for n in hits.values()),
        "redundant_successful_contacts": sum(n - 1 for n in hits.values()),
        "repeated_successful_pairs": sum(n >= 2 for n in pairs.values()),
        "redundant_parallel_contacts": sum(n - 1 for n in pairs.values()),
    }
    assert successful - distinct == result["redundant_successful_contacts"]
    return result


def old_hits_pmf(r, q, intensity):
    """PMF of Bernoulli(q) + Poisson(intensity*q)."""
    return float((1 - q) * poisson.pmf(r, intensity * q) + q * poisson.pmf(r - 1, intensity * q))


def exact_categories(q, intensity):
    hits = [old_hits_pmf(r, q, intensity) for r in range(4)]
    return {
        "old_successful_contacts": hits + [max(0.0, 1 - sum(hits))],
        "redundant_successful_contacts": [
            hits[0] + hits[1],
            hits[2],
            hits[3],
            max(0.0, 1 - sum(hits)),
        ],
        "old_multihit_recipients": [hits[0] + hits[1], max(0.0, 1 - hits[0] - hits[1])],
    }


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="New directory; existing results are never overwritten",
    )
    parser.add_argument("--replicates", type=int, default=18086)
    parser.add_argument("--seed", type=int, default=20261019)
    parser.add_argument("--p", type=float, default=0.5)
    parser.add_argument("--intensity", type=float, default=0.5)
    parser.add_argument("--max-contacts", type=int, default=10000)
    args = parser.parse_args()
    if args.replicates < 1 or not 0 <= args.p <= 1 or not np.isfinite(args.intensity) or args.intensity <= 0:
        parser.error("replicates must be positive; p in [0,1]; intensity finite and positive")
    if args.max_contacts < 2 or args.seed < 0:
        parser.error("max-contacts >= 2 and seed >= 0 required")
    if args.output_dir.exists():
        parser.error("output directory already exists; choose a new path")
    root = Path(__file__).resolve().parents[1]
    args.output_dir.mkdir(parents=True)
    initial = nx.MultiDiGraph()
    initial.add_nodes_from([(0, {"message": True}), (1, {"message": False})])
    initial.add_edges_from([(0, 0), (1, 1)])
    model = Graph(
        c=0,
        p=args.p,
        delta_in=1,
        delta_out=1,
        lambda_func=lambda k, c: args.intensity,
        rng=np.random.default_rng(args.seed),
    ).initialize(initial, 2, 1)
    rows = []
    with (args.output_dir / "batches.jsonl").open("w") as stream:
        for replicate in range(args.replicates):
            edges, _ = model.sample_step_edges(current_round=1, k=args.max_contacts)
            # Sampling does not change G0.
            metrics = contact_metrics(edges, {0, 1}, {0})
            rows.append({"replicate": replicate, "batch_size": len(edges), **metrics})
            stream.write(
                json.dumps(
                    {
                        "replicate": replicate,
                        "edges": [[int(u), int(v)] for u, v in edges],
                    }
                )
                + "\n"
            )
    with (args.output_dir / "metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    q = (1 - args.p) / 4
    theory = exact_categories(q, args.intensity)
    specs = [
        (
            "old_successful_contacts",
            ["0", "1", "2", "3", ">=4"],
            "Successful contacts to old recipients",
        ),
        (
            "redundant_successful_contacts",
            ["0", "1", "2", ">=3"],
            "Extra successful contacts per batch",
        ),
        ("old_multihit_recipients", ["0", "1"], "Old nodes reached at least twice"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.5), layout="constrained")
    frequencies = []
    for ax, (key, labels, title) in zip(axes, specs):
        cap = len(labels) - 1
        counts = np.bincount([min(row[key], cap) for row in rows], minlength=len(labels))
        empirical = counts / args.replicates
        ci = np.array([binomtest(int(n), args.replicates).proportion_ci(method="wilson") for n in counts])
        x = np.arange(len(labels))
        ax.bar(x - 0.18, theory[key], 0.34, color="#225ea8", label="Exact one-step law")
        ax.bar(x + 0.18, empirical, 0.34, color="#fd8d3c", label="Simulation")
        ax.errorbar(
            x + 0.18,
            empirical,
            yerr=[empirical - ci[:, 0], ci[:, 1] - empirical],
            fmt="none",
            ecolor="#444444",
            capsize=3,
        )
        ax.set(
            xticks=x,
            xticklabels=labels,
            title=title,
            xlabel="Count in one batch",
            ylabel="Probability",
            ylim=(0, 1.12),
        )
        for pos, count, probability in zip(x, counts, empirical):
            ax.text(
                pos,
                max(probability, theory[key][pos]) + 0.025,
                f"n={count}",
                ha="center",
                fontsize=8,
            )
        ax.grid(axis="y", alpha=0.2)
        for label, count, exact, observed, interval in zip(labels, counts, theory[key], empirical, ci):
            frequencies.append(
                {
                    "metric": key,
                    "category": label,
                    "count": int(count),
                    "exact_probability": exact,
                    "empirical_probability": float(observed),
                    "wilson_95_lower": float(interval[0]),
                    "wilson_95_upper": float(interval[1]),
                }
            )
    axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle(
        f"Repeated successful pull contacts: two initial nodes, one informed; p={args.p:g}, lambda={args.intensity:g}\n"
        f"{args.replicates:,} independent one-step batches; error bars: pointwise 95% Wilson intervals",
        fontsize=11,
    )
    fig.savefig(args.output_dir / "repeated-successful-contacts.png", dpi=220)
    fig.savefig(args.output_dir / "repeated-successful-contacts.pdf")
    plt.close(fig)
    with (args.output_dir / "frequencies.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(frequencies[0]))
        writer.writeheader()
        writer.writerows(frequencies)
    totals = {key: sum(row[key] for row in rows) for key in rows[0] if key != "replicate"}
    affected = sum(row["redundant_successful_contacts"] > 0 for row in rows)
    source_hashes = {
        name: digest(root / name)
        for name in [
            "main.py",
            "propagation.py",
            "scripts/repeated_successful_contacts.py",
        ]
    }
    summary = {
        "parameters": {**vars(args), "output_dir": str(args.output_dir)},
        "initial_graph": {"nodes": [0, 1], "edges": [[0, 0], [1, 1]], "informed": [0]},
        "rule": "target_to_source; frozen initial graph and informed set; one batch; old edges do not transmit",
        "q_old_recipient": q,
        "totals": totals,
        "batches_with_repeated_success": affected,
        "exact_probability_of_repeated_success": theory["old_multihit_recipients"][1],
        "exact_mean_extra_contacts": (1 + args.intensity) * q - (1 - (1 - q) * np.exp(-args.intensity * q)),
        "poisson_truncation_tail_per_batch": float(poisson.sf(args.max_contacts - 1, args.intensity)),
        "exact_law": "K=B(q)+Pois(lambda*q); R=max(K-1,0); H=1{K>=2}. Exact columns use the untruncated law.",
        "limitations": "One step and one old susceptible node only. Counts are within batches, not across trajectories. No claim about 200-step ensembles.",
        "source_sha256": source_hashes,
        "output_sha256": {p.name: digest(p) for p in sorted(args.output_dir.iterdir()) if p.is_file()},
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(
        json.dumps(
            {
                "totals": totals,
                "affected_batches": affected,
                "output_dir": str(args.output_dir),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
