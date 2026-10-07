import networkx as nx
import numpy as np

from main import Graph


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
