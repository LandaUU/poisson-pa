from __future__ import annotations

import numpy as np
import scipy as sp
import networkx as nx
from dataclasses import dataclass
from typing import Callable, Optional


def lambda_f(k: int, c: float) -> float:
    return float(np.power(k, c) * np.log(8 * k))


@dataclass(slots=True)
class GraphParams:
    c: float
    p: float
    delta_in: int
    delta_out: int
    node_limit_in_p_case: bool  # legacy option


class Graph:
    """
    Directed Poisson-PA graph evolution (Wang–Resnick style):

    Per round t = 1..:
      - sample ΔM_t in {1..k} with parameter λ(t)
      - for each of ΔM_t edges:
          with prob p:   add edge (new_node -> v), v ~ in-degree+δ_in
          with prob 1-p: add edge (u -> v), u ~ out-degree+δ_out, v ~ in-degree+δ_in (independent)

    Message spreading (your “2nd variant”):
      for each new edge (w -> v): if v is informed and w is not, then w becomes informed.
    """

    def __init__(
        self,
        c: float,
        p: float,
        delta_in: int,
        delta_out: int,
        node_limit_in_p_case: bool = True,
        lambda_func: Callable[[int, float], float] = lambda_f,
        rng: Optional[np.random.Generator] = None,
    ):
        self.params = GraphParams(
            c=c,
            p=p,
            delta_in=delta_in,
            delta_out=delta_out,
            node_limit_in_p_case=node_limit_in_p_case,
        )
        self.lambda_func = lambda_func
        self.rng = rng if rng is not None else np.random.default_rng()

        self._graph: nx.MultiDiGraph = nx.MultiDiGraph()
        self._round: int = 0
        self._next_node_id: int = 0

        # fast message bookkeeping
        self._messaged: set[int] = set()

        # histories (post-round snapshots)
        self.nodes_history: list[int] = []
        self.informed_history: list[int] = []
        self.new_informed_history: list[int] = []

    @property
    def G(self) -> nx.MultiDiGraph:
        return self._graph

    @property
    def round(self) -> int:
        return self._round

    def initialize(
        self,
        initial_state: Optional[nx.MultiDiGraph] = None,
        initial_message_nodes: Optional[set[int]] = None,
    ) -> Graph:
        if initial_state is None:
            self._graph = nx.MultiDiGraph()
            self._graph.add_node(0, message=True)
            self._graph.add_edge(0, 0)
            self._messaged = {0}
            self._next_node_id = 1
        else:
            self._graph = initial_state
            if initial_message_nodes is None:
                # infer from attributes if present
                self._messaged = {
                    n
                    for n, has_msg in self._graph.nodes(data="message")
                    if bool(has_msg)
                }
            else:
                self._messaged = set(initial_message_nodes)

            # assumes integer node ids; if sequential, this is correct
            self._next_node_id = (
                (max(self._graph.nodes) + 1) if self._graph.number_of_nodes() else 0
            )

            # keep node attrs consistent
            for n in self._graph.nodes:
                self._graph.nodes[n]["message"] = n in self._messaged

        self._round = 0
        self._snapshot_histories(new_informed=0)
        return self

    def _lambda(self, t: int) -> float:
        return self.lambda_func(t, self.params.c)

    def _delta_m_weights(self, t: int, k: int) -> np.ndarray:
        """
        Weights for ΔM_t in {1..k} proportional to P(Poisson(λ(t)) = m-1).
        Normalization is done explicitly for numerical stability.
        """
        lam = self._lambda(t)
        m = np.arange(1, k + 1, dtype=np.float64)

        # log P(X=m-1) = -λ + (m-1)log λ - log((m-1)!)
        logw = -lam + (m - 1.0) * np.log(lam) - sp.special.gammaln(m)
        logw -= np.max(logw)
        w = np.exp(logw)
        w /= w.sum()
        return w

    def _in_out_probs(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns (p_in, p_out) over current node ids [0..n-1].
        Assumes node ids are consecutive ints, which holds for this construction.
        """
        n = self._graph.number_of_nodes()
        if n == 0:
            raise RuntimeError("Graph is empty; call initialize().")

        # degrees in MultiDiGraph are ints; convert to float arrays
        in_deg = np.fromiter(
            (d for _, d in self._graph.in_degree(range(n))), dtype=np.float64, count=n
        )
        out_deg = np.fromiter(
            (d for _, d in self._graph.out_degree(range(n))), dtype=np.float64, count=n
        )

        pin = in_deg + float(self.params.delta_in)
        pout = out_deg + float(self.params.delta_out)

        pin_sum = pin.sum()
        pout_sum = pout.sum()

        # Defensive: in case deltas are zero and graph has 0 degrees (shouldn't after init)
        if pin_sum <= 0 or pout_sum <= 0:
            raise RuntimeError("Non-positive normalization in degree probabilities.")

        pin /= pin_sum
        pout /= pout_sum
        return pin, pout

    def _add_new_node(self) -> int:
        nid = self._next_node_id
        self._next_node_id += 1
        self._graph.add_node(nid, message=False)
        return nid

    def _propagate_messages_on_edges(self, edges: list[tuple[int, int]]) -> int:
        """
          for edge (w -> v): if v is informed and w is not, then w becomes informed.
        Returns number of newly informed nodes in this round.
        """
        new_informed = 0
        for w, v in edges:
            if (v in self._messaged) and (w not in self._messaged):
                self._messaged.add(w)
                # keep nx attrs in sync for drawing
                self._graph.nodes[w]["message"] = True
                new_informed += 1
        return new_informed

    def _snapshot_histories(self, new_informed: int) -> None:
        self.nodes_history.append(self._graph.number_of_nodes())
        self.informed_history.append(len(self._messaged))
        self.new_informed_history.append(new_informed)

    def step(self, k: int, verbose: bool = False) -> Graph:
        """
        Do one evolution round.
        k is the truncation upper bound for ΔM_t (ΔM_t ∈ {1..k}).
        """
        self._round += 1

        # sample ΔM_t
        w = self._delta_m_weights(t=self._round, k=k)
        delta_m = int(self.rng.choice(np.arange(1, k + 1), p=w))
        if verbose:
            print(f"Delta M_{self._round} = {delta_m}")

        # snapshot probs at round start (per model)
        p_in, p_out = self._in_out_probs()
        n0 = self._graph.number_of_nodes()

        edges: list[tuple[int, int]] = []

        # legacy option: if True, all p-edges in this round share a single new node
        shared_new_node: Optional[int] = None
        if self.params.node_limit_in_p_case:
            shared_new_node = self._add_new_node()

        # generate edges
        # Fast path: independent sampling for beta-case; no O(n^2) pair list.
        for _ in range(delta_m):
            if self.rng.random() < self.params.p:
                # p-case: new node -> v, v ~ p_in
                if shared_new_node is not None:
                    w_node = shared_new_node
                else:
                    w_node = self._add_new_node()

                v = int(self.rng.choice(n0, p=p_in))
                edges.append((w_node, v))
            else:
                # beta-case: u -> v, u ~ p_out, v ~ p_in (independent)
                u = int(self.rng.choice(n0, p=p_out))
                v = int(self.rng.choice(n0, p=p_in))
                edges.append((u, v))

        self._graph.add_edges_from(edges)

        new_inf = self._propagate_messages_on_edges(edges)
        self._snapshot_histories(new_informed=new_inf)
        return self

    def evolute(self, k: int, rounds: int, verbose: bool = False) -> Graph:
        for _ in range(rounds):
            self.step(k=k, verbose=verbose)
        return self

    def draw(self, title: Optional[str] = None, path: Optional[str] = None) -> None:
        import matplotlib.pyplot as plt

        fig = plt.figure("Graph", figsize=(4, 4), clear=True)
        ax = fig.add_subplot(1, 1, 1)

        pos = nx.spring_layout(
            self._graph, iterations=5000
        )  # lower default, still OK visually
        nodes = list(self._graph.nodes())

        node_colors = ["#2d2d2d" if (n in self._messaged) else "#a4a2a2" for n in nodes]

        nx.draw_networkx(
            self._graph,
            pos,
            ax=ax,
            node_color=node_colors,
            font_color="white",
            edge_color="gray",
            edgecolors="white",
            width=1,
            with_labels=False,
            node_size=80,
            arrows=True,
        )
        ax.set_axis_off()
        if title:
            ax.set_title(title)
        fig.tight_layout()

        if path is None:
            plt.show()
        else:
            plt.savefig(path)

    def draw_informed_fraction(self) -> None:
        """
        Plots |S_t| / N_t where:
          N_t = total nodes after round t
          |S_t| = total informed nodes after round t
        """
        import matplotlib.pyplot as plt

        N = np.array(self.nodes_history, dtype=np.float64)
        S = np.array(self.new_informed_history, dtype=np.float64)
        frac = np.divide(S, N, out=np.zeros_like(S), where=(N > 0))

        plt.figure(figsize=(5, 2.7), layout="constrained")
        plt.plot(np.arange(len(frac)), frac)
        plt.ylabel("|S_t| / N_t")
        plt.xlabel("t")
        plt.title("Informed fraction over rounds")
        plt.show()


def init_graph():
    _g = nx.MultiDiGraph()

    _g.add_edge(0, 0)

    _g.add_edge(1, 1)

    _g.nodes[0]["message"] = True

    return _g


g2 = Graph(
    c=0.1, p=0.99, delta_in=1, delta_out=1, node_limit_in_p_case=False
).initialize(initial_message_nodes={0})

g2.evolute(k=10000, rounds=500, verbose=True)

# g2.draw_informed_fraction()
g2.draw_informed_fraction()
