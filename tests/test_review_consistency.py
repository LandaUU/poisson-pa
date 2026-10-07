from collections import defaultdict
from itertools import product

import networkx as nx
import numpy as np
import pytest

from main import Graph


def test_exact_two_contact_law_counts_vertices_not_contacts():
    # Reviewer 3 counterexample: two equal-weight old nodes, only 0 informed.
    categories = [("new", None, 0, 0.25), ("new", None, 1, 0.25)]
    categories += [("old", u, v, 0.125) for u in range(2) for v in range(2)]
    law = defaultdict(float)
    for pair in product(categories, repeat=2):
        initial = nx.MultiDiGraph([(0, 0), (1, 1)])
        initial.nodes[0]["message"] = True
        graph = Graph(0.2, 0.5, 1, 1).initialize(initial, 2, 1)
        batch = []
        births = 0
        probability = 1.0
        for kind, u, v, weight in pair:
            if kind == "new":
                u = 2 + births
                births += 1
            batch.append((u, v))
            probability *= weight
        graph._graph.add_edges_from(batch)
        graph._simple_message_distribution(graph._graph, batch)
        law[births, graph.S[-1]] += probability
    distribution = {
        delta: sum(value for (_, informed), value in law.items() if informed == delta) for delta in range(3)
    }
    assert sum(law.values()) == pytest.approx(1)
    assert distribution == pytest.approx({0: 0.390625, 1: 0.484375, 2: 0.125})
    assert distribution[2] != pytest.approx(0.375**2)  # The rejected binomial-contact shortcut.
    for births, expected in enumerate((0.25, 0.5, 0.25)):
        assert sum(value for (count, _), value in law.items() if count == births) == pytest.approx(expected)


def test_arbitrary_G0_normalization_and_frozen_batch_weights(monkeypatch):
    initial = nx.MultiDiGraph([(0, 1), (0, 1), (1, 1), (2, 0), (2, 1)])
    initial.nodes[0]["message"] = True
    graph = Graph(0.2, 0.5, 2, 3).initialize(initial, 3, 1)
    pin, pout = graph._get_in_out_probs()
    np.testing.assert_allclose(pin, np.array([3.0, 6.0, 2.0]) / 11)
    np.testing.assert_allclose(pout, np.array([5.0, 4.0, 5.0]) / 14)
    captured = []

    class ControlledRNG:
        draws = iter((0.1, 0.9, 0.1, 0.9))

        def random(self):
            return next(self.draws)

        def choice(self, n, p):
            captured.append((n, p.copy()))
            return 0

    graph.rng = ControlledRNG()
    monkeypatch.setattr(graph, "sample_delta_m", lambda **kwargs: 4)
    contacts, births = graph.sample_step_edges(1, 10)
    assert contacts == [(3, 0), (0, 0), (4, 0), (0, 0)] and births == 2
    assert graph._graph.number_of_nodes() == 3 and graph._graph.number_of_edges() == 5
    assert all(n == 3 for n, _ in captured)
    for (_, actual), expected in zip(captured, (pin, pout, pin, pin, pout, pin)):
        np.testing.assert_array_equal(actual, expected)
