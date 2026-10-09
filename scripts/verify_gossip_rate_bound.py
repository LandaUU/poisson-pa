"""Independent finite-state checks of the fixed-topology completion bound."""

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.stats import binom


def recipients(pair, state, protocol):
    u, v = pair
    result = set()
    if protocol in ("pull", "pp") and v in state:
        result.add(u)
    if protocol in ("push", "pp") and u in state:
        result.add(v)
    return result - state


def verify_all(batch_sizes=(1, 2, 3)):
    if not batch_sizes or any(m not in (1, 2, 3) for m in batch_sizes):
        raise ValueError("Batch sizes must be chosen from 1, 2, 3")
    vertices = set(range(3))
    candidates = [(u, v) for u in vertices for v in vertices if u != v]
    states = [set(c) for size in (1, 2) for c in itertools.combinations(vertices, size)]
    index = {frozenset(s): i for i, s in enumerate(states)}
    checks, max_ratio = 0, 0.0
    for mask in range(1 << len(candidates)):
        edges = [edge for i, edge in enumerate(candidates) if mask & (1 << i)]
        for loops in (False, True):
            pairs = edges + ([(u, u) for u in vertices] if loops else [])
            degree = {u: sum(a == u for a, b in pairs) for u in vertices}
            active = sum(d > 0 for d in degree.values())
            if not pairs:
                continue
            weights = [1 / (active * degree[u]) for u, v in pairs]
            for protocol in ("pull", "push", "pp"):
                q = [
                    sum(w for pair, w in zip(pairs, weights, strict=True) if recipients(pair, s, protocol))
                    for s in states
                ]
                q_min = min(q)
                if q_min == 0:
                    continue
                for m in batch_sizes:
                    transition = np.zeros((6, 6))
                    alpha = 1 - (1 - q_min) ** m
                    for i, s in enumerate(states):
                        drift = 0.0
                        for sequence in itertools.product(range(len(pairs)), repeat=m):
                            probability = np.prod([weights[j] for j in sequence])
                            additions = set().union(*(recipients(pairs[j], s, protocol) for j in sequence))
                            drift += probability * len(additions)
                            target = s | additions
                            if target != vertices:
                                transition[i, index[frozenset(target)]] += probability
                        formula = sum(
                            1
                            - (
                                1
                                - sum(
                                    w
                                    for pair, w in zip(pairs, weights, strict=True)
                                    if x in recipients(pair, s, protocol)
                                )
                            )
                            ** m
                            for x in vertices - s
                        )
                        if abs(drift - formula) >= 1e-12:
                            raise AssertionError("Conditional drift disagrees with independent enumeration")
                    expectations = np.linalg.solve(np.eye(6) - transition, np.ones(6))
                    tail = np.ones(6)
                    for i, s in enumerate(states):
                        upper = (3 - len(s)) / alpha
                        if expectations[i] > upper + 1e-10:
                            raise AssertionError("Exact mean exceeds completion bound")
                        max_ratio = max(max_ratio, expectations[i] / upper)
                    for t in range(13):
                        for i, s in enumerate(states):
                            if tail[i] > binom.cdf(3 - len(s) - 1, t, alpha) + 1e-10:
                                raise AssertionError("Exact tail exceeds binomial bound")
                        tail = transition @ tail
                    checks += 1
    return dict(
        passed=True,
        graph_protocol_batch_cases=checks,
        maximum_exact_mean_to_bound_ratio=max_ratio,
        batch_sizes=list(batch_sizes),
        scope="all 3-vertex directed simple edge sets, with and without all self-loops; positive minimum cuts; all incomplete nonempty states; 13 tail checkpoints",
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-file", type=Path, default=Path("results/gossip-rate-verification.json"))
    parser.add_argument("--batch-sizes", default="1,2,3")
    args = parser.parse_args(argv)
    if args.output_file.exists():
        raise FileExistsError(args.output_file)
    report = verify_all(tuple(int(x) for x in args.batch_sizes.split(",")))
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    args.output_file.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
