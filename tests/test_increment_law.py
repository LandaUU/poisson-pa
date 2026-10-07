"""Exact checks for reply step 2; no dependency on manuscript checkout."""

import itertools
import math
from collections import defaultdict

import networkx as nx
import numpy as np
import pytest
from scipy.stats import poisson

from main import Graph


def compositions(total, count):
    if count == 1:
        yield (total,)
    else:
        for first in range(total + 1):
            for rest in compositions(total - first, count - 1):
                yield (first, *rest)


def fixed_contacts(alpha, beta, qs, contacts):
    """Multinomial enumeration, independent of the generating function."""
    probabilities = (alpha, beta, *qs, 1 - alpha - beta - sum(qs))
    law = defaultdict(float)
    for counts in compositions(contacts, len(probabilities)):
        probability = math.factorial(contacts)
        for count, category in zip(counts, probabilities):
            probability *= category**count / math.factorial(count)
        if probability == 0:
            continue
        births = counts[0] + counts[1]
        informed = counts[0] + sum(count > 0 for count in counts[2:-1])
        law[births, informed] += probability
    return dict(law)


def poisson_contacts(alpha, beta, qs, intensity, max_contacts=16):
    law = defaultdict(float)
    for contacts in range(1, max_contacts + 1):
        weight = poisson.pmf(contacts - 1, intensity)
        for state, probability in fixed_contacts(alpha, beta, qs, contacts).items():
            law[state] += weight * probability
    return dict(law), float(poisson.sf(max_contacts - 1, intensity))


def pgf(z, t, alpha, beta, qs, intensity):
    factors = [math.exp(-intensity * q) + (1 - math.exp(-intensity * q)) * t for q in qs]
    bracket = (1 - alpha - beta - sum(qs) + alpha * z * t + beta * z) * math.prod(factors)
    bracket += sum(q * t * math.prod(factors[:idx] + factors[idx + 1 :]) for idx, q in enumerate(qs))
    return math.exp(intensity * (alpha * (z * t - 1) + beta * (z - 1))) * bracket


def increment_mean(alpha, qs, intensity):
    return (1 + intensity) * alpha + sum(1 - (1 - q) * math.exp(-intensity * q) for q in qs)


def run_checks():
    examples = []
    scenarios = [
        (0.25, 0.25, (0.125,)),
        (0.2, 0.3, (0.1, 0.15)),
        (0, 0, (0.3, 0.2)),
        (0.4, 0.6, ()),
        (1, 0, ()),
        (0, 0, ()),
    ]
    for alpha, beta, qs in scenarios:
        for intensity in (0, 0.5, 1):
            law, tail = poisson_contacts(alpha, beta, qs, intensity)
            assert abs(sum(law.values()) - (1 - tail)) < 2e-14
            for z, t in itertools.product((0, 0.2, 0.7, 1), repeat=2):
                enumerated = sum(w * z**b * t**r for (b, r), w in law.items())
                assert abs(enumerated - pgf(z, t, alpha, beta, qs, intensity)) <= tail + 3e-14
            mean = sum(r * w for (b, r), w in law.items())
            # D <= M; the omitted first moment is explicitly bounded.
            moment_tail = poisson.sf(15, intensity) + intensity * poisson.sf(14, intensity)
            assert abs(mean - increment_mean(alpha, qs, intensity)) <= moment_tail + 3e-14
            examples.append(
                dict(
                    alpha=alpha,
                    beta=beta,
                    qs=qs,
                    intensity=intensity,
                    total_mass=sum(law.values()),
                    tail_mass=tail,
                    mean=mean,
                    exact_mean=increment_mean(alpha, qs, intensity),
                )
            )
    return examples


def test_joint_pgf_and_mean_against_independent_enumeration():
    scenarios = run_checks()
    assert len(scenarios) == 18
    for case in scenarios:
        alpha, beta, qs, intensity = (case[k] for k in ("alpha", "beta", "qs", "intensity"))
        a = alpha + sum(qs)
        assert pgf(1, 1, alpha, beta, qs, intensity) == pytest.approx(1)
        assert pgf(1, 0, alpha, beta, qs, intensity) == pytest.approx((1 - a) * math.exp(-intensity * a))
        p = alpha + beta
        for z in (0, 0.2, 0.7, 1):
            assert pgf(z, 1, alpha, beta, qs, intensity) == pytest.approx(
                (1 - p + p * z) * math.exp(intensity * p * (z - 1))
            )


@pytest.mark.parametrize("p", [0, 0.5, 1])
def test_three_steps_with_full_graph_history(p):
    """Exact rational graph-state enumeration; weights change after each edge."""
    from collections import defaultdict
    from fractions import Fraction

    p = Fraction(p)
    states = {(2, ((0, 0), (1, 1)), (0,)): Fraction(1)}
    accumulated = Fraction(1)
    means = [Fraction(1)]
    increments = []
    previous_cdf = {i: Fraction(int(i >= 1)) for i in range(6)}
    for step in range(1, 4):
        following = defaultdict(Fraction)
        mean_increment = Fraction(0)
        for (n, edges, informed), mass in states.items():
            incoming = [sum(v == u for _, v in edges) + 1 for u in range(n)]
            outgoing = [sum(v == u for v, _ in edges) + 1 for u in range(n)]
            total = len(edges) + n
            H = sum(Fraction(incoming[u], total) for u in informed)
            mu = p * H + (1 - p) * H * sum(Fraction(outgoing[u], total) for u in range(n) if u not in informed)
            mean_increment += mass * mu
            options = []
            for target in range(n):
                options.append((n, target, p * Fraction(incoming[target], total), True))
                for source in range(n):
                    options.append(
                        (
                            source,
                            target,
                            (1 - p) * Fraction(outgoing[source] * incoming[target], total**2),
                            False,
                        )
                    )
            for source, target, probability, new in options:
                if not probability:
                    continue
                updated = set(informed)
                if target in informed:
                    updated.add(source)
                key = (
                    n + int(new),
                    tuple(sorted((*edges, (source, target)))),
                    tuple(sorted(updated)),
                )
                # Compare independent enumeration with the production updater.
                initial = nx.MultiDiGraph()
                initial.add_nodes_from(range(n))
                initial.add_edges_from(edges)
                for u in informed:
                    initial.nodes[u]["message"] = True
                graph = Graph(0, float(p), 1, 1, lambda_func=lambda n, c: 0).initialize(initial, n, len(informed))
                graph._graph.add_edge(source, target)
                graph._simple_message_distribution(graph._graph, [(source, target)])
                actual_set = tuple(
                    sorted(u for u, attrs in graph._graph.nodes(data=True) if attrs.get("message", False))
                )
                assert actual_set == key[2]
                assert graph.S[-1] == len(updated) - len(informed)
                following[key] += mass * probability
        assert sum(following.values()) == 1
        accumulated += mean_increment
        actual = sum(len(informed) * mass for (n, edges, informed), mass in following.items())
        assert actual == accumulated
        means.append(actual)
        increments.append(mean_increment)
        for i in range(6):
            cdf = sum(mass for (n, edges, informed), mass in following.items() if len(informed) <= i)
            assert cdf <= previous_cdf[i]
            previous_cdf[i] = cdf
        states = following
    # Both sides describe min(K, 3), so no unchecked infinite tail is discarded.
    for theta in (0, 0.7, 2):
        mixture = sum(float(means[k]) * poisson.pmf(k, theta) for k in range(3))
        mixture += float(means[3]) * poisson.sf(2, theta)
        drift = 1 + sum(float(mu) * poisson.sf(j - 1, theta) for j, mu in enumerate(increments, 1))
        assert mixture == pytest.approx(drift, abs=2e-14)


def test_zero_lambda_one_contact_and_full_information():
    initial = nx.MultiDiGraph([(0, 0), (1, 1)])
    for node in initial:
        initial.nodes[node]["message"] = True
    graph = Graph(0, 1, 1, 1, lambda_func=lambda n, c: 0, rng=np.random.default_rng(7)).initialize(initial, 2, 2)
    assert graph._get_probabilities(0, 0, 1).tolist() == [1]
    graph.evolute(3, k=5)
    assert graph.N == [2, 1, 1, 1]
    assert graph.S_total == [2, 3, 4, 5]
    assert graph._graph.number_of_edges() == 5


@pytest.mark.parametrize("intensity", [-1, float("nan"), float("inf")])
def test_invalid_lambda(intensity):
    graph = Graph(0, 0.5, 1, 1, lambda_func=lambda n, c: intensity)
    with pytest.raises(ValueError, match="finite and nonnegative"):
        graph.sample_delta_m(1, 10)


def test_invalid_truncation():
    graph = Graph(0, 0.5, 1, 1, lambda_func=lambda n, c: 0)
    with pytest.raises(ValueError, match="positive"):
        graph.sample_delta_m(1, 0)
