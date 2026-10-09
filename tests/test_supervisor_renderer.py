"""CLI integration without manuscript paths, external data or saved model ensembles."""

import csv
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.render_supervisor_revision import Renderer, main, parse_args


def write_csv(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)


@pytest.fixture
def saved_inputs(tmp_path):
    source = tmp_path / "inputs"
    nk = source / "nk/p_0.5_c_0.2_runs_4_steps_3"
    counts = np.array([[2, 2, 3, 4], [2, 3, 4, 5], [2, 4, 6, 8], [2, 3, 5, 6]])
    write_csv(
        nk / "nk_trajectories.csv",
        [dict(run_id=i + 1, step=k, N_k=int(value)) for i, run in enumerate(counts) for k, value in enumerate(run)],
    )
    write_csv(
        nk / "nk_summary.csv",
        [dict(step=k, empirical_mean=float(value)) for k, value in enumerate(counts.mean(axis=0))],
    )
    (nk / "config.json").write_text(json.dumps(dict(n0=2, runs=4, steps=3, p=0.5, c=0.2)))
    write_csv(
        source / "sk/p_0.5_c_0.2_runs_4_steps_3/sk_trajectories.csv",
        [
            dict(run_id=i + 1, step=k, ratio=0.5 if k == 0 else min(1, 0.5 + i * k / 20))
            for i in range(4)
            for k in range(4)
        ],
    )
    write_csv(
        source / "pn/pn_snapshot_rows.csv", [dict(p_theory=0.5, p_empirical=0.5, successes=250, replications=500)]
    )
    protocols = ("pull", "push", "push-pull", "uniform-pull", "uniform-push", "uniform-push-pull")
    write_csv(
        source / "gossip/metrics.csv",
        [
            dict(scenario=dataset, protocol=protocol, final_coverage=0.5, mean_all_positive_budgets=0.4)
            for dataset in ("CollegeMsg", "email-Eu-core", "MathOverflow-a2q")
            for protocol in protocols
        ],
    )
    for dataset in ("CollegeMsg", "email-Eu-core", "MathOverflow-a2q"):
        write_csv(
            source / "gossip" / dataset / "calendar-curves.csv",
            [dict(unix_hour=k, **{p: 0.5 for p in protocols}) for k in (0, 1)],
        )
    for folder, filename in (
        ("collegemsg_fit", "CollegeMsg.txt.gz"),
        ("email_eu_core_temporal_fit", "email-Eu-core-temporal.txt.gz"),
        ("mathoverflow_a2q_fit", "sx-mathoverflow-a2q.txt.gz"),
    ):
        path = source / "data" / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(path, "wt") as handle:
            handle.write("# fixture\n0 1 0\n1 0 1\n0 0 2\n")
        summary = source / "fitted" / folder / "fit_summary.json"
        summary.parent.mkdir(parents=True, exist_ok=True)
        summary.write_text(
            json.dumps(dict(config=dict(data_path="data/" + filename), fit=dict(iota_in_hat=2, iota_out_hat=2)))
        )
        write_csv(
            summary.with_name("simulated_message_trajectories.csv"),
            [
                dict(run_id=i + 1, message_rule="source_to_target", step=k, ratio=0.5 + i * k / 10)
                for i in range(2)
                for k in range(2)
            ],
        )
    return source


def arguments(source, output):
    return [
        "--output-dir",
        str(output),
        "--nk-dir",
        str(source / "nk"),
        "--sk-dir",
        str(source / "sk"),
        "--pn-dir",
        str(source / "pn"),
        "--fitted-root",
        str(source / "fitted"),
        "--data-root",
        str(source / "data"),
        "--gossip-dir",
        str(source / "gossip"),
        "--nk-runs",
        "4",
        "--sk-runs",
        "4",
        "--steps",
        "3",
        "--p-values",
        "0.5",
        "--c-values",
        "0.2",
        "--bootstrap-resamples",
        "100",
        "--gossip-runs",
        "4",
    ]


def test_all_sections_from_explicit_inputs_and_provenance(saved_inputs, tmp_path):
    output = tmp_path / "rendered"
    args = arguments(saved_inputs, output) + [
        "--sections",
        "node-count,local-probability,coverage,angles,gossip,calendar,archived-fitted",
    ]
    main(args)
    manifest = json.loads((output / "manifest.json").read_text())
    assert len(list(output.glob("*.pdf"))) == 6
    assert len(list(output.glob("*.png"))) == 6
    assert manifest["settings"]["bootstrap_resamples"] == 100
    assert "not replay-validated" in manifest["scope"]
    for path, digest in manifest["input_sha256"].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest
    with (output / "local-probability-ci.csv").open() as handle:
        record = next(csv.DictReader(handle))
    assert float(record["ci_low"]) < 0.5 < float(record["ci_high"])
    original = (output / "manifest.json").read_bytes()
    with pytest.raises(FileExistsError):
        main(args)
    assert (output / "manifest.json").read_bytes() == original


def test_missing_input_fails_before_creating_output(tmp_path):
    output = tmp_path / "never-created"
    with pytest.raises(FileNotFoundError, match="Missing required inputs"):
        main(["--output-dir", str(output), "--sections", "local-probability", "--pn-dir", str(tmp_path / "missing")])
    assert not output.exists()


def test_duplicate_and_incomplete_trajectories_rejected(tmp_path):
    renderer = Renderer(parse_args(["--output-dir", str(tmp_path / "out")]))
    path = tmp_path / "bad.csv"
    write_csv(path, [dict(run_id=1, step=0, ratio=0.5)] * 2)
    with pytest.raises(ValueError, match="Duplicate"):
        renderer.matrix(path, "ratio")
    write_csv(path, [dict(run_id=1, step=0, ratio=0.5), dict(run_id=2, step=1, ratio=0.5)])
    with pytest.raises(ValueError, match="missing"):
        renderer.matrix(path, "ratio")


def test_unknown_or_unconfigured_sections_rejected(tmp_path):
    for section in ("unrecognized", "gossip", "calendar"):
        with pytest.raises(SystemExit):
            parse_args(["--output-dir", str(tmp_path / "out"), "--sections", section])
