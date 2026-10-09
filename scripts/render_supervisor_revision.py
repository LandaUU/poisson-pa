"""Render conditional Monte Carlo estimates from explicitly supplied saved inputs.

No graph simulation or network-level statistical inference is performed here.
Run as `python -m scripts.render_supervisor_revision --help`.
"""

import argparse
import csv
import hashlib
import json
import os
import platform
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/poisson-pa-empirical-render")

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy

from scripts.empirical_intervals import (
    bootstrap_variance_interval,
    hoeffding_radius,
    student_mean_interval,
    wilson_interval,
)


class Renderer:
    def __init__(self, args):
        self.args = args
        self.inputs = {}
        self.p_values = tuple(float(x) for x in args.p_values.split(","))
        self.c_values = tuple(float(x) for x in args.c_values.split(","))
        if not self.p_values or not self.c_values or len(self.p_values) > 3 or len(self.c_values) > 3:
            raise ValueError("Supply between one and three p-values and c-values")
        if len(set(self.p_values)) != len(self.p_values) or len(set(self.c_values)) != len(self.c_values):
            raise ValueError("Grid parameters must be distinct")
        if not all(np.isfinite(x) and 0 <= x <= 1 for x in self.p_values) or not all(
            np.isfinite(x) and x > 0 for x in self.c_values
        ):
            raise ValueError("Require p in [0,1] and positive finite c")
        self.representative = (
            (0.5, 0.2) if 0.5 in self.p_values and 0.2 in self.c_values else (self.p_values[0], self.c_values[0])
        )

    def record(self, path):
        self.inputs[str(path.resolve())] = hashlib.sha256(path.read_bytes()).hexdigest()
        return path

    def rows(self, path):
        with self.record(path).open() as handle:
            result = list(csv.DictReader(handle))
        if not result:
            raise ValueError(f"Empty input CSV: {path}")
        return result

    def matrix(self, path, field):
        records = self.rows(path)
        runs = sorted({int(r["run_id"]) for r in records})
        steps = sorted({int(r["step"]) for r in records})
        result = np.full((len(runs), len(steps)), np.nan)
        run_idx = {r: i for i, r in enumerate(runs)}
        step_idx = {s: i for i, s in enumerate(steps)}
        for row in records:
            i, j = run_idx[int(row["run_id"])], step_idx[int(row["step"])]
            if not np.isnan(result[i, j]):
                raise ValueError("Duplicate run/step in trajectory CSV")
            result[i, j] = float(row[field])
        if not np.isfinite(result).all():
            raise ValueError("Trajectory CSV contains missing or nonfinite values")
        if steps != list(range(len(steps))):
            raise ValueError("Trajectory steps must be contiguous from zero")
        return np.array(steps), result

    def save(self, fig, name):
        fig.savefig(self.args.output_dir / f"{name}.pdf", bbox_inches="tight")
        fig.savefig(self.args.output_dir / f"{name}.png", dpi=180, bbox_inches="tight")
        plt.close(fig)

    def write_csv(self, name, records):
        with (self.args.output_dir / name).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)

    def growth(self):
        base = self.args.nk_dir
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), layout="constrained")
        final, checkpoints = [], []
        rng = np.random.default_rng(self.args.bootstrap_seed)
        for p in self.p_values:
            for c in self.c_values:
                folder = base / f"p_{p:g}_c_{c:g}_runs_{self.args.nk_runs}_steps_{self.args.steps}"
                x, data = self.matrix(folder / "nk_trajectories.csv", "N_k")
                config = json.loads(self.record(folder / "config.json").read_text())
                if data.shape != (self.args.nk_runs, self.args.steps + 1):
                    raise ValueError("Node-count trajectory shape disagrees with --nk-runs/--steps")
                mean, mean_low, mean_high = student_mean_interval(data)
                half = mean_high - mean
                var, low, high = bootstrap_variance_interval(data, rng=rng, resamples=self.args.bootstrap_resamples)
                intensity = np.zeros_like(x, dtype=float)
                intensity[1:] = x[1:] ** c * np.log(8 * x[1:])
                theory_mean = config["n0"] + p * (x + np.cumsum(intensity))
                theory_var = x * p * (1 - p) + p * np.cumsum(intensity)
                if not np.allclose(mean, [float(r["empirical_mean"]) for r in self.rows(folder / "nk_summary.csv")]):
                    raise ValueError("Archived node-count means disagree with trajectory CSV")
                final.append(
                    dict(
                        p=p,
                        c=c,
                        mean=mean[-1],
                        mean_low=mean[-1] - half[-1],
                        mean_high=mean[-1] + half[-1],
                        variance=var[-1],
                        variance_low=low[-1],
                        variance_high=high[-1],
                    )
                )
                if (p, c) == self.representative:
                    for ax, estimate, a, b, theory, ylabel in (
                        (axes[0], mean, mean - half, mean + half, theory_mean, "Mean node count"),
                        (axes[1], var, low, high, theory_var, "Node-count variance"),
                    ):
                        ax.fill_between(x, a, b, color="tab:blue", alpha=0.3, label="Pointwise 95% CI")
                        ax.plot(x, estimate, color="tab:blue", label="Empirical estimate", linewidth=1.3)
                        ax.plot(x, theory, color="black", linestyle="--", label="Theory", linewidth=1.2)
                        ax.set(xlabel="Evolution step", ylabel=ylabel)
                        ax.legend(fontsize=8)
                    checkpoints = [
                        dict(
                            step=int(k),
                            mean=mean[k],
                            mean_low=mean[k] - half[k],
                            mean_high=mean[k] + half[k],
                            variance=var[k],
                            variance_low=low[k],
                            variance_high=high[k],
                        )
                        for k in x
                    ]
        self.save(fig, "node-count-ci")
        self.write_csv("node-count-final-ci.csv", final)
        self.write_csv("node-count-curve-ci.csv", checkpoints)
        lines = [
            r"\begin{table}[pos=tbp]\centering\small",
            r"\caption{Final node-count estimates with pointwise 95\% conditional Monte Carlo intervals. Mean: Student intervals; variance: whole-trajectory percentile bootstrap ("
            + str(self.args.bootstrap_resamples)
            + r" resamples).}",
            r"\label{tab:nk-ci}\begin{tabular}{ccrr}\toprule",
            r"$p$ & $c$ & Mean [95\% CI] & Variance [95\% CI] \\\midrule",
        ]
        for r in final:
            lines.append(
                f"{r['p']} & {r['c']} & {r['mean']:.2f} [{r['mean_low']:.2f}, {r['mean_high']:.2f}] & {r['variance']:.2f} [{r['variance_low']:.2f}, {r['variance_high']:.2f}] "
                + r"\\"
            )
        lines.append(r"\bottomrule\end{tabular}\end{table}")
        (self.args.output_dir / "node-count-table.tex").write_text("\n".join(lines) + "\n")

    def local_probability(self):
        base = self.args.pn_dir
        records = self.rows(base / "pn_snapshot_rows.csv")
        fig, ax = plt.subplots(figsize=(6.3, 4.5), layout="constrained")
        out = []
        for r in records:
            n, successes = int(r["replications"]), int(r["successes"])
            low, high = wilson_interval(successes, n)
            if not np.isclose(float(r["p_empirical"]), successes / n, rtol=0, atol=1e-12):
                raise ValueError("Saved success frequency disagrees with binomial counts")
            out.append({**r, "ci_low": low, "ci_high": high})
            ax.plot([float(r["p_theory"])] * 2, [low, high], color="tab:blue", alpha=0.22, linewidth=0.8)
        ax.scatter(
            [float(r["p_theory"]) for r in records], [float(r["p_empirical"]) for r in records], s=10, color="tab:blue"
        )
        ax.plot([0, 1], [0, 1], "k--", linewidth=1)
        ax.set(
            xlabel="Theoretical local success probability",
            ylabel="Empirical probability (pointwise 95% Wilson CI)",
            xlim=(0, 1),
            ylim=(0, 1),
        )
        self.save(fig, "local-probability-ci")
        self.write_csv("local-probability-ci.csv", out)

    def coverage(self):
        base = self.args.sk_dir
        fig, ax = plt.subplots(figsize=(8.6, 5.0), layout="constrained")
        inset = ax.inset_axes([0.57, 0.18, 0.40, 0.30])
        fig2, axes = plt.subplots(1, 2, figsize=(9, 3.6), layout="constrained")
        mean_grid, radius_grid = (
            np.zeros((len(self.p_values), len(self.c_values))),
            np.zeros((len(self.p_values), len(self.c_values))),
        )
        curves, finals = [], []
        colors, styles, markers = ("tab:blue", "tab:orange", "tab:green"), ("-", "--", ":"), ("o", "s", "^")
        radius = hoeffding_radius(self.args.sk_runs)
        for i, p in enumerate(self.p_values):
            for j, c in enumerate(self.c_values):
                x, data = self.matrix(
                    base / f"p_{p:g}_c_{c:g}_runs_{self.args.sk_runs}_steps_{self.args.steps}/sk_trajectories.csv",
                    "ratio",
                )
                if data.shape != (self.args.sk_runs, self.args.steps + 1) or not ((data >= 0) & (data <= 1)).all():
                    raise ValueError("Coverage trajectories disagree with --sk-runs/--steps or leave [0,1]")
                mean, sd = data.mean(axis=0), data.std(axis=0, ddof=1)
                low, high = np.maximum(0, mean - radius), np.minimum(1, mean + radius)
                low[0] = high[0] = mean[0]
                for target in [ax, inset] if p == 0.9 else [ax]:
                    target.fill_between(x, low, high, color=colors[i], alpha=0.035)
                    target.plot(
                        x,
                        mean,
                        color=colors[i],
                        linestyle=styles[j],
                        marker=markers[j],
                        markevery=35,
                        markersize=3.2,
                        linewidth=1.3,
                        label=f"p={p}, c={c}",
                    )
                for k in x:
                    curves.append(dict(p=p, c=c, step=int(k), mean=mean[k], ci_low=low[k], ci_high=high[k]))
                finals.append(dict(p=p, c=c, mean=mean[-1], sd=sd[-1], ci_low=low[-1], ci_high=high[-1]))
                mean_grid[i, j], radius_grid[i, j] = mean[-1], radius
        ax.set(xlabel="Evolution step", ylabel="Mean relative coverage", ylim=(0, 1.025))
        ax.legend(ncol=3, fontsize=7.5, loc="lower left")
        if 0.9 in self.p_values and self.args.steps > 25:
            inset.set(xlim=(25, self.args.steps), ylim=(0.35, 1.0), title="p = 0.9")
        else:
            inset.set_visible(False)
        inset.tick_params(labelsize=7)
        self.save(fig, "coverage-color-ci")
        for target, data, title in zip(
            axes, (mean_grid, radius_grid), ("Mean final coverage", "95% Hoeffding interval radius"), strict=True
        ):
            im = target.imshow(data, vmin=0, vmax=1, cmap="viridis")
            target.set(
                xticks=range(len(self.c_values)),
                xticklabels=self.c_values,
                yticks=range(len(self.p_values)),
                yticklabels=self.p_values,
                xlabel="c",
                ylabel="p",
                title=title,
            )
            for i in range(len(self.p_values)):
                for j in range(len(self.c_values)):
                    target.text(
                        j,
                        i,
                        f"{data[i, j]:.4f}",
                        ha="center",
                        va="center",
                        color="white" if data[i, j] < 0.5 else "black",
                    )
            fig2.colorbar(im, ax=target, shrink=0.8)
        self.save(fig2, "coverage-final-ci")
        self.write_csv("coverage-curve-ci.csv", curves)
        self.write_csv("coverage-final-ci.csv", finals)
        lines = [
            r"\begin{table}[pos=tbp]\centering\small",
            r"\caption{Archived final coverage: mean, trajectory SD, and pointwise 95\% Hoeffding interval for the mean ("
            + str(self.args.sk_runs)
            + r" independent trajectories). These archived spreading runs have not been independently replayed under the corrected batch rule.}",
            r"\label{tab:sk-ratio-final}\begin{tabular}{ccrrr}\toprule",
            r"$p$ & $c$ & Mean & SD & 95\% CI \\\midrule",
        ]
        for r in finals:
            lines.append(
                f"{r['p']} & {r['c']} & {r['mean']:.4f} & {r['sd']:.4f} & [{r['ci_low']:.4f}, {r['ci_high']:.4f}] "
                + r"\\"
            )
        lines.append(r"\bottomrule\end{tabular}\end{table}")
        (self.args.output_dir / "coverage-table.tex").write_text("\n".join(lines) + "\n")

    def angles(self):
        # Recompute descriptive angular samples; no iid-vertex inference is asserted.
        import gzip
        from collections import Counter

        fig, ax = plt.subplots(figsize=(7.5, 4.0), layout="constrained")
        summaries, sample_rows = [], []
        for label, folder, color in (
            ("CollegeMsg", "collegemsg_fit", "tab:blue"),
            ("email-Eu-core", "email_eu_core_temporal_fit", "tab:orange"),
            ("MathOverflow a2q", "mathoverflow_a2q_fit", "tab:green"),
        ):
            summary = json.loads(self.record(self.args.fitted_root / folder / "fit_summary.json").read_text())
            path = self.record(self.args.data_root / Path(summary["config"]["data_path"]).name)
            incoming, outgoing = Counter(), Counter()
            with gzip.open(path, "rt") as handle:
                for line in handle:
                    if not line.strip() or line.startswith("#"):
                        continue
                    u, v, *_ = line.split()
                    outgoing[u] += 1
                    incoming[v] += 1
            nodes = sorted(set(incoming) | set(outgoing))
            iv = np.array([incoming[n] for n in nodes], dtype=float)
            ov = np.array([outgoing[n] for n in nodes], dtype=float)
            radius = iv + ov
            selected = radius >= np.quantile(radius[radius > 0], 0.995)
            # Tail-index estimates correspond to the archived calibration in reply.
            fit = summary.get("fit", summary)
            a = fit["iota_in_hat"] / fit["iota_out_hat"]
            power = iv[selected] ** a
            samples = np.sort(power / (power + ov[selected]))
            ax.step(
                np.r_[0, samples, 1],
                np.r_[0, np.arange(1, len(samples) + 1) / len(samples), 1],
                where="post",
                label=f"{label} (m={len(samples)})",
                color=color,
            )
            summaries.append(dict(dataset=label, count=len(samples), a=a, quantile=0.995))
            sample_rows.extend(dict(dataset=label, angle=float(v)) for v in samples)
        ax.set(
            xlabel=r"Angular coordinate $\Theta$", ylabel="Empirical cumulative fraction", xlim=(0, 1), ylim=(0, 1.02)
        )
        ax.legend(fontsize=8)
        self.save(fig, "angular-empirical")
        self.write_csv("angular-samples.csv", sample_rows)
        (self.args.output_dir / "angular-summary.json").write_text(json.dumps(summaries, indent=2) + "\n")

    def gossip_intervals(self):
        records = self.rows(self.args.gossip_dir / "metrics.csv")
        radius = hoeffding_radius(self.args.gossip_runs)
        lines = [
            r"\begin{table}[pos=tbp]\centering\small",
            r"\caption{Conditional uniform-gossip means [pointwise 95\% Hoeffding intervals] on observed growth, "
            + str(self.args.gossip_runs)
            + r" repetitions. F: final coverage; A: mean coverage over every positive budget. Recorded-contact protocols are deterministic conditional on the dataset.}",
            r"\label{tab:observed-gossip-ci}\begin{tabular}{llrrr}\toprule",
            r"Dataset & Metric & U-pull & U-push & U-P--P \\\midrule",
        ]
        output = []
        for dataset in ("CollegeMsg", "email-Eu-core", "MathOverflow-a2q"):
            for metric, field in (("F", "final_coverage"), ("A", "mean_all_positive_budgets")):
                cells = []
                for protocol in ("uniform-pull", "uniform-push", "uniform-push-pull"):
                    row = next(r for r in records if r["scenario"] == dataset and r["protocol"] == protocol)
                    value = float(row[field])
                    low, high = max(0, value - radius), min(1, value + radius)
                    cells.append(f"{value:.3f} [{low:.3f}, {high:.3f}]")
                    output.append(
                        dict(dataset=dataset, protocol=protocol, metric=metric, mean=value, ci_low=low, ci_high=high)
                    )
                lines.append(" & ".join([dataset, metric, *cells]) + r"\\")
        lines.append(r"\bottomrule\end{tabular}\end{table}")
        (self.args.output_dir / "observed-gossip-ci.tex").write_text("\n".join(lines) + "\n")
        self.write_csv("observed-gossip-ci.csv", output)

    def calendar_intervals(self):
        protocols = ("pull", "push", "push-pull", "uniform-pull", "uniform-push", "uniform-push-pull")
        fig, axes = plt.subplots(1, 3, figsize=(10, 4.5), layout="constrained")
        radius = hoeffding_radius(self.args.gossip_runs)
        for ax, dataset in zip(axes, ("CollegeMsg", "email-Eu-core", "MathOverflow-a2q"), strict=True):
            records = self.rows(self.args.gossip_dir / dataset / "calendar-curves.csv")
            hours = np.array([float(r["unix_hour"]) for r in records])
            days = (hours - hours[0]) / 24
            for protocol, color in zip(
                protocols, ("tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown"), strict=True
            ):
                mean = np.array([float(r[protocol]) for r in records])
                ax.plot(days, mean, label=protocol, color=color, linewidth=1)
                if protocol.startswith("uniform-"):
                    ax.fill_between(
                        days, np.maximum(0, mean - radius), np.minimum(1, mean + radius), color=color, alpha=0.10
                    )
            ax.set(title=dataset, xlabel="Days after suffix start", ylim=(0, 1.02))
        axes[0].set_ylabel("Mean coverage")
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside lower center", ncol=3, fontsize=9)
        self.save(fig, "observed-calendar-ci")

    def archived_fitted_intervals(self):
        final = []
        for label, folder in (("CollegeMsg", "collegemsg_fit"), ("email-Eu-core", "email_eu_core_temporal_fit")):
            records = self.rows(self.args.fitted_root / folder / "simulated_message_trajectories.csv")
            terminal = {}
            for row in records:
                if row["message_rule"] != "source_to_target":
                    continue
                run = int(row["run_id"])
                if run not in terminal or int(row["step"]) > int(terminal[run]["step"]):
                    terminal[run] = row
            if len(terminal) < 2:
                raise ValueError("At least two fitted trajectories are required")
            mean = np.mean([float(r["ratio"]) for r in terminal.values()])
            radius = hoeffding_radius(len(terminal))
            final.append(
                dict(
                    dataset=label,
                    runs=len(terminal),
                    mean=mean,
                    ci_low=max(0, mean - radius),
                    ci_high=min(1, mean + radius),
                )
            )
        lines = [
            r"\begin{table}[pos=tbp]\centering\small",
            r"\caption{Archived push fitted-coverage means with pointwise 95\% Hoeffding intervals, conditional on the earlier calibration and update implementation. These intervals exclude parameter uncertainty and do not validate the corrected pull model.}",
            r"\label{tab:archived-fitted-ci}\begin{tabular}{lrr}\toprule",
            r"Dataset & Mean & 95\% CI \\\midrule",
        ]
        for r in final:
            lines.append(f"{r['dataset']} & {r['mean']:.4f} & [{r['ci_low']:.4f}, {r['ci_high']:.4f}] " + r"\\")
        lines.append(r"\bottomrule\end{tabular}\end{table}")
        (self.args.output_dir / "archived-fitted-ci.tex").write_text("\n".join(lines) + "\n")
        self.write_csv("archived-fitted-ci.csv", final)


SECTIONS = {
    "node-count": "growth",
    "local-probability": "local_probability",
    "coverage": "coverage",
    "angles": "angles",
    "gossip": "gossip_intervals",
    "calendar": "calendar_intervals",
    "archived-fitted": "archived_fitted_intervals",
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, required=True, help="New output directory; existing paths are rejected"
    )
    parser.add_argument(
        "--sections",
        default="node-count,local-probability,coverage",
        help="Comma-separated sections: " + ",".join(SECTIONS),
    )
    parser.add_argument(
        "--nk-dir", type=Path, default=Path("results/nk/grid/grid_runs_1000_steps_200_p_0.1-0.5-0.9_c_0.1-0.2-0.3")
    )
    parser.add_argument(
        "--sk-dir", type=Path, default=Path("results/sk/grid/grid_runs_100_steps_200_p_0.1-0.5-0.9_c_0.1-0.2-0.3")
    )
    parser.add_argument(
        "--pn-dir", type=Path, default=Path("results/pn/g0_1/base_runs_5_repl_500_p_0.1-0.5-0.9_c_0.1-0.2-0.3")
    )
    parser.add_argument("--fitted-root", type=Path, default=Path("results"))
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument(
        "--gossip-dir", type=Path, help="External matched-gossip archive containing metrics.csv and calendar CSVs"
    )
    parser.add_argument("--p-values", default="0.1,0.5,0.9")
    parser.add_argument("--c-values", default="0.1,0.2,0.3")
    parser.add_argument("--nk-runs", type=int, default=1000)
    parser.add_argument("--sk-runs", type=int, default=100)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20261010)
    parser.add_argument("--gossip-runs", type=int, default=200)
    args = parser.parse_args(argv)
    args.selected_sections = args.sections.split(",")
    if not args.selected_sections or any(x not in SECTIONS for x in args.selected_sections):
        parser.error("Unknown section; choose from " + ",".join(SECTIONS))
    if len(set(args.selected_sections)) != len(args.selected_sections):
        parser.error("Sections must be distinct")
    if any(x in args.selected_sections for x in ("gossip", "calendar")) and args.gossip_dir is None:
        parser.error("--gossip-dir is required for gossip/calendar sections")
    if min(args.nk_runs, args.sk_runs, args.bootstrap_resamples) < 2 or args.steps < 1 or args.gossip_runs < 1:
        parser.error("Require >=2 trajectory/bootstrap runs, steps>=1, gossip-runs>=1")
    return args


def main(argv=None):
    args = parse_args(argv)
    renderer = Renderer(args)
    # Validate every requested file before creating the output directory.
    required = []
    for p in renderer.p_values:
        for c in renderer.c_values:
            if "node-count" in args.selected_sections:
                folder = args.nk_dir / f"p_{p:g}_c_{c:g}_runs_{args.nk_runs}_steps_{args.steps}"
                required.extend(folder / name for name in ("nk_trajectories.csv", "nk_summary.csv", "config.json"))
            if "coverage" in args.selected_sections:
                required.append(
                    args.sk_dir / f"p_{p:g}_c_{c:g}_runs_{args.sk_runs}_steps_{args.steps}/sk_trajectories.csv"
                )
    if "local-probability" in args.selected_sections:
        required.append(args.pn_dir / "pn_snapshot_rows.csv")
    if "gossip" in args.selected_sections:
        required.append(args.gossip_dir / "metrics.csv")
    if "calendar" in args.selected_sections:
        required.extend(
            args.gossip_dir / d / "calendar-curves.csv" for d in ("CollegeMsg", "email-Eu-core", "MathOverflow-a2q")
        )
    if "archived-fitted" in args.selected_sections:
        required.extend(
            args.fitted_root / d / "simulated_message_trajectories.csv"
            for d in ("collegemsg_fit", "email_eu_core_temporal_fit")
        )
    if "angles" in args.selected_sections:
        for d in ("collegemsg_fit", "email_eu_core_temporal_fit", "mathoverflow_a2q_fit"):
            summary = args.fitted_root / d / "fit_summary.json"
            required.append(summary)
            if summary.is_file():
                metadata = json.loads(summary.read_text())
                required.append(args.data_root / Path(metadata["config"]["data_path"]).name)
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        raise FileNotFoundError("Missing required inputs:\n" + "\n".join(missing))
    if any(x in args.selected_sections for x in ("gossip", "calendar")):
        archive_manifest = args.gossip_dir / "manifest.json"
        if archive_manifest.is_file():
            metadata = json.loads(renderer.record(archive_manifest).read_text())
            if metadata.get("runs") != args.gossip_runs:
                raise ValueError("--gossip-runs disagrees with the source archive manifest")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    for name in args.selected_sections:
        getattr(renderer, SECTIONS[name])()
    settings = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    manifest = dict(
        settings=settings,
        input_sha256=renderer.inputs,
        source_sha256={
            Path(__file__).name: hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "empirical_intervals.py": hashlib.sha256(
                Path(__file__).with_name("empirical_intervals.py").read_bytes()
            ).hexdigest(),
        },
        versions=dict(
            python=platform.python_version(),
            numpy=np.__version__,
            scipy=scipy.__version__,
            matplotlib=matplotlib.__version__,
        ),
        ci_level=0.95,
        scope="conditional Monte Carlo; archived spreading not replay-validated; no parameter or observed-network uncertainty",
    )
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(args.output_dir)


if __name__ == "__main__":
    main()
