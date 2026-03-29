import itertools
import matplotlib
import networkx as nx
import numpy as np
import scipy as sp
from dataclasses import dataclass
import matplotlib.pyplot as plt

from typing import Self, Optional, Callable, Iterable, Sequence

matplotlib.use("TkAgg")


def lambda_f(k, c):
    return np.power(k, c) * np.log(8 * k)


def lambda_const(k, c):
    return 10


@dataclass(slots=True)
class GraphParams:
    c: float
    p: float
    delta_in: int
    delta_out: int


class Graph:
    def __init__(
        self,
        c: float,
        p: float,
        delta_in: int,
        delta_out: int,
        lambda_func: Callable[[int, float], float] = lambda_f,
        rng: Optional[np.random.Generator] = None,
    ):
        self.params = GraphParams(
            c=c,
            p=p,
            delta_in=delta_in,
            delta_out=delta_out,
        )
        self.lambda_func = lambda_func
        self.rng = rng if rng is not None else np.random.default_rng()

        self.N: list[int] = []
        self.S: list[int] = []

        self.N_total = []
        self.S_total = []

    def initialize(
        self,
        initial_state: Optional[nx.MultiDiGraph] = None,
        initial_N: Optional[int] = None,
        initial_S: Optional[int] = None,
    ) -> Self:
        if initial_state:
            self._graph = initial_state
            assert initial_N
            assert initial_S
            self.N.append(initial_N)
            self.S.append(initial_S)
            self.N_total.append(initial_N)
            self.S_total.append(initial_S)
        else:
            self._graph = nx.MultiDiGraph()
            self._graph.add_node(0, message=True)
            self._graph.add_edge(0, 0)
            self.N.append(1)
            self.S.append(1)
            self.N_total.append(1)
            self.S_total.append(1)

        self._round = 0

        return self

    def _get_probabilities(self, n: int, c: float, k: int):
        lambda_val = self.lambda_func(n + 1, c)
        ks = np.arange(1, k + 1)

        log_probs = -lambda_val + (ks - 1) * np.log(lambda_val) - sp.special.gammaln(ks)

        weights = np.exp(log_probs - np.max(log_probs))
        return weights

    def _get_in_out_probs(self) -> tuple[np.ndarray, np.ndarray]:
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

        if pin_sum <= 0 or pout_sum <= 0:
            raise RuntimeError("Non-positive normalization in degree probabilities.")

        pin /= pin_sum
        pout /= pout_sum
        return pin, pout

    def _simple_message_distribution(
        self, G: nx.MultiDiGraph, edges: list[tuple[int, int]]
    ) -> None:
        """В случае, если передача сообщения не происходит далее по цепочке,
        то можно использовать эту функцию"""

        self.S.append(0)

        for w, v in edges:
            if G.nodes[v].get("message") and not G.nodes[w].get("message"):
                G.nodes[w]["message"] = True
                self.S[-1] += 1

    def evolute(self, rounds: int, k: int = 10**5) -> Self:
        for current_round in range(self._round + 1, self._round + rounds + 1):
            K_weights = self._get_probabilities(
                n=current_round - 1, c=self.params.c, k=k
            )
            M = self.rng.choice(k, p=K_weights / K_weights.sum()) + 1
            print(f"Delta M_{current_round} = {M}")

            p_in, p_out = self._get_in_out_probs()
            node_pointer = self._graph.number_of_nodes()
            n0 = self._graph.number_of_nodes()
            edges = []
            self.N.append(0)
            for _ in range(M):
                if self.rng.random() < self.params.p:
                    # alpha case (no loops)

                    node_2 = self.rng.choice(n0, p=p_in)

                    edge = (node_pointer, node_2)

                    edges.append(edge)

                    node_pointer += 1

                    self.N[-1] += 1
                else:
                    # beta case
                    w, v = (
                        self.rng.choice(n0, p=p_out),
                        self.rng.choice(n0, p=p_in),
                    )
                    edges.append((w, v))

            self._graph.add_edges_from(edges)

            self._simple_message_distribution(self._graph, edges)

            self.N_total.append(self.N_total[-1] + self.N[-1])
            self.S_total.append(self.S_total[-1] + self.S[-1])

        self._round += rounds
        return self

    def draw(self, title: Optional[str] = None):
        fig = plt.figure("Graph", figsize=(4, 4), clear=True)
        # Create a gridspec for adding subplots of different sizes
        axgrid = fig.add_gridspec(1, 1)

        ax0 = fig.add_subplot(axgrid[0])
        pos = nx.spring_layout(
            self._graph,
            iterations=5000,
            # scale=5,
            k=(1 / (self._graph.number_of_nodes() ** 0.5)) + 1,
            fixed=list(self._graph.nodes)[0:2],
            pos={0: (0, 0), 1: (2, 2)},
        )
        nodes = list(self._graph.nodes())
        node_colors = [
            "#2d2d2d" if self._graph.nodes[n].get("message", False) else "#a4a2a2"
            for n in nodes
        ]

        nx.draw_networkx(
            self._graph,
            pos,
            ax=ax0,
            node_color=node_colors,
            font_color="white",
            edge_color="gray",
            edgecolors="white",
            width=1,
            with_labels=False,
            node_size=80,
            arrows=True,
        )
        ax0.set_axis_off()

        fig.tight_layout()
        if title:
            ax0.set_title(title)
        # plt.show()
        plt.savefig(f"./images/{self.params.p}_{self._round}.png")

    @staticmethod
    def draw_ns_statistic(
        series: Iterable[tuple[Sequence[float], Sequence[float], str]],
        *,
        title: str = "",
        xlabel: str = "k",
        ylabel: str = "# S_k/N_k",
        filename: str = "",
    ) -> None:
        """
        Draw multiple (N_total, S_total) series on a single figure with black lines and save to file.
        series: iterable of (N_total, S_total, label)
        """
        fig, ax = plt.subplots(figsize=(6, 3.2), layout="constrained")
        line_styles = ["-", "--", ":", "-."]

        for i, (N_total, S_total, label) in enumerate(series):
            N = np.asarray(N_total, dtype=np.float64)
            S = np.asarray(S_total, dtype=np.float64)
            frac = np.divide(
                S, N, out=np.zeros_like(S, dtype=np.float64), where=(N > 0)
            )

            ax.plot(
                list(range(len(frac))),
                frac,
                color="black",
                linewidth=1.2,
                label=label,
                linestyle=line_styles[i % len(line_styles)],
            )

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid()
        ax.legend()
        if filename:
            plt.savefig(f"./images/{filename}.png")
        else:
            plt.show()


def init_graph():
    _g = nx.MultiDiGraph()

    _g.add_edge(0, 0)

    _g.add_edge(1, 1)

    _g.nodes[0]["message"] = True

    return _g


def _avg_totals_over_runs(
    *,
    make_graph: Callable[[], Graph],
    evolute_steps: int,
    k: int,
    runs: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Runs 'runs' independent simulations, returns (N_total_avg, S_total_avg).
    Assumes each graph exposes:
      - .evolute(rounds: int, k: int) -> self
      - .N_total: Sequence[float]
      - .S_total: Sequence[float]
    """
    N_sum: Optional[np.ndarray] = None
    S_sum: Optional[np.ndarray] = None

    for _ in range(runs):
        g = make_graph()
        g.evolute(rounds=evolute_steps, k=k)

        Nt = np.asarray(g.N_total, dtype=np.float64)
        St = np.asarray(g.S_total, dtype=np.float64)

        if N_sum is None:
            N_sum = Nt.copy()
            S_sum = St.copy()
        else:
            if len(N_sum) != len(Nt):
                raise ValueError(
                    f"Length mismatch across runs: got {len(Nt)} but expected {len(N_sum)}. "
                    "Ensure each run produces the same number of recorded points."
                )
            N_sum += Nt
            S_sum += St

    assert N_sum is not None and S_sum is not None
    return N_sum / runs, S_sum / runs


# --- experiment config ---
RUNS = 20
EVOLUTE_STEPS = 500
K = 10_000

# Overlay these c values on one figure
C_VALUES = [0.1, 0.2, 0.3]
P_VALUES = [0.1, 0.5, 0.9, 0.99]

DELTA_IN = 1
DELTA_OUT = 1


for P in P_VALUES:
    # --- build all series ---
    series = []
    for c in C_VALUES:

        def make_graph_for_c(c_=c):
            # Replace the following with your actual initialization:
            g = Graph(
                c=c,
                p=P,
                delta_in=1,
                delta_out=1,
            ).initialize(init_graph(), initial_N=2, initial_S=1)
            return g

        N_avg, S_avg = _avg_totals_over_runs(
            make_graph=make_graph_for_c,
            evolute_steps=EVOLUTE_STEPS,
            k=K,
            runs=RUNS,
        )
        series.append((N_avg, S_avg, f"c={c}"))

    # --- save plot ---
    Graph.draw_ns_statistic(series, filename=f"{P}_many_c_{RUNS}_stat")


# ROUNDS = 20
#
# c = 0.4
# p = 0.99
#
#
# N_TOTAL: Optional[np.ndarray] = None
# S_TOTAL: Optional[np.ndarray] = None
#
# for i in range(ROUNDS):
#     g = Graph(
#         c=c,
#         p=p,
#         delta_in=1,
#         delta_out=1,
#     ).initialize(init_graph(), initial_N=2, initial_S=1)
#
#     g.evolute(rounds=500)
#
#     if N_TOTAL is None:
#         N_TOTAL = g.N_total
#     else:
#         min_len = min(len(N_TOTAL), len(g.N_total))
#         N_TOTAL = np.array(N_TOTAL[0:min_len]) + np.array(g.N_total[0:min_len])
#
#     if S_TOTAL is None:
#         S_TOTAL = g.S_total
#     else:
#         min_len = min(len(S_TOTAL), len(g.S_total))
#         S_TOTAL = np.array(S_TOTAL[0:min_len]) + np.array(g.S_total[0:min_len])
#
# N_TOTAL = N_TOTAL / ROUNDS
# S_TOTAL = S_TOTAL / ROUNDS
#
# Graph.draw_ns_statistic(N_TOTAL, S_TOTAL, filename=f"{p}_{c}_{ROUNDS}_stat")

# C = 0.1
#
# g0 = Graph(
#     c=C,
#     p=0.1,
#     delta_in=1,
#     delta_out=1,
# ).initialize(init_graph(), initial_N=2, initial_S=1)
# g0.draw()
#
# g0.evolute(rounds=1).draw()
#
# g0.evolute(rounds=4).draw()
#
# g0.evolute(rounds=5).draw()
#
#
# g1 = Graph(
#     c=C,
#     p=0.5,
#     delta_in=1,
#     delta_out=1,
# ).initialize(init_graph(), initial_N=2, initial_S=1)
# g1.draw()
#
# g1.evolute(rounds=1).draw()
#
# g1.evolute(rounds=4).draw()
#
# g1.evolute(rounds=5).draw()
#
# g2 = Graph(
#     c=C,
#     p=0.9,
#     delta_in=1,
#     delta_out=1,
# ).initialize(init_graph(), initial_N=2, initial_S=1)
# g2.draw()
#
# g2.evolute(rounds=1).draw()
#
# g2.evolute(rounds=4).draw()
#
# g2.evolute(rounds=5).draw()
#
# g3 = Graph(
#     c=C,
#     p=0.99,
#     delta_in=1,
#     delta_out=1,
# ).initialize(init_graph(), initial_N=2, initial_S=1)
# g3.draw()
#
# g3.evolute(rounds=1).draw()
#
# g3.evolute(rounds=4).draw()
#
# g3.evolute(rounds=5).draw()
