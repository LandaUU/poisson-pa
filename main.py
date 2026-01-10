import itertools
import networkx as nx
import numpy as np
import scipy as sp
from dataclasses import dataclass
from random import choices, randint, seed
import matplotlib.pyplot as plt
# from networkx.drawing.nx_agraph import graphviz_layout

from typing import Self, Optional, Literal, Callable


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
    msg_distr: Literal["from", "to"]
    orientation: Literal["directed", "reversed"]
    node_creation_p_case: bool


class Graph:
    def __init__(
        self,
        c: float,
        p: float,
        delta_in: int,
        delta_out: int,
        msg_distr: Literal["from", "to"] = "from",
        orientation: Literal["directed", "reversed"] = "directed",
        node_creation_p_case: bool = False,
        lambda_func: Callable[[int, float], float] = lambda_f,
    ):
        self.params = GraphParams(
            c=c,
            p=p,
            delta_in=delta_in,
            delta_out=delta_out,
            msg_distr=msg_distr,
            orientation=orientation,
            node_creation_p_case=node_creation_p_case,
        )
        self.lambda_func = lambda_func

    def initialize(self, initial_state: Optional[nx.MultiDiGraph] = None) -> Self:
        seed(randint(10**2, 10**7))

        if initial_state:
            self._graph = initial_state
        else:
            self._graph = nx.MultiDiGraph()
            self._graph.add_node(0, message=True)
            self._graph.add_edge(0, 0)

        self._evolution_rounds = 0

        return self

    def _get_probabilities(self, n: int, c: float, k: int):
        lambda_val = self.lambda_func(n + 1, c)
        ks = np.arange(1, k + 1)

        log_probs = -lambda_val + (ks + 1) * np.log(lambda_val) - sp.special.gammaln(ks)

        weights = np.exp(log_probs - np.max(log_probs))
        return weights

    def _get_probabilities_in(
        self, G: nx.MultiDiGraph, delta_in: int
    ) -> np.typing.NDArray:
        return np.array(
            [(G.in_degree(n) + delta_in) for n in range(G.number_of_nodes())]
        ).dot(1 / (sum(t[1] for t in G.in_degree()) + G.number_of_nodes() * delta_in))

    def _get_probabilities_out(
        self, G: nx.MultiDiGraph, delta_out: int
    ) -> np.typing.NDArray:
        return np.array(
            [(G.out_degree(n) + delta_out) for n in range(G.number_of_nodes())]
        ).dot(1 / (sum(t[1] for t in G.out_degree()) + G.number_of_nodes() * delta_out))

    def evolute(self, k: int, rounds: int) -> Self:
        for current_round in range(
            self._evolution_rounds + 1, self._evolution_rounds + rounds + 1
        ):
            K = self._get_probabilities(
                n=self._graph.number_of_nodes(), c=self.params.c, k=k
            )
            M = choices(population=list(range(1, k + 1)), weights=K.tolist())[0]
            print(f"Delta M_{current_round} = {M}")

            probabilities_in = self._get_probabilities_in(
                self._graph, self.params.delta_in
            )
            probabilities_out = self._get_probabilities_out(
                self._graph, self.params.delta_out
            )
            graph_nodes = set(self._graph.nodes)
            nodes = list(itertools.product(graph_nodes, graph_nodes))
            print("Edges pool: ", nodes)
            probabilities_final = np.multiply(
                [probabilities_in[n1] for n1, _ in nodes],
                [probabilities_out[n2] for _, n2 in nodes],
            )
            last_node = self._graph.number_of_nodes() - 1
            edges = []
            node_pointer = last_node

            def p_case():
                return (
                    choices(
                        population=(0, 1),
                        weights=(self.params.p, 1 - self.params.p),
                    )[0]
                    == 0
                )

            def loops_case():
                """В графе только петли и при этом мы уже прошли 1-stage => не пускаем в (1-p)"""
                return (
                    not [t for t in nodes if t[0] != t[1]]
                    and self._evolution_rounds > 1
                )

            node_pointer = self._graph.number_of_nodes()

            for _ in range(M):
                if p_case() or loops_case():
                    # alpha case (no loops)

                    node_2 = choices(
                        population=list(range(self._graph.number_of_nodes())),
                        weights=probabilities_in.tolist(),
                    )[0]

                    edge = (node_pointer, node_2)

                    edges.append(edge)

                    # Если у нас одновременное добавление, то вершины не вставляются последовательно
                    if self.params.node_creation_p_case:
                        node_pointer += 1
                else:
                    # beta case (deny loops on 1 stage)
                    edge = choices(
                        population=list(nodes),
                        weights=probabilities_final.tolist(),
                    )[0]
                    edges.append(edge)
            # В prob_final у нас пары вероятностей (in, out) => (v, w)
            for v, w in edges:
                if self.params.orientation == "reversed":
                    # swap #1
                    v, w = w, v

                self._graph.add_edge(w, v)

                if self.params.msg_distr == "to":
                    # Делаем swap для нужного нам типа распространения сообщения
                    # from - сообщение передается от исходящего узла к входящему
                    # to - от входящего к исходящему
                    w, v = v, w

                if self._graph.nodes[w].get("message"):
                    self._graph.nodes[v]["message"] = True

        self._evolution_rounds += rounds
        return self

    def draw(self, title: Optional[str] = None):
        print(nx.dfs_tree(self._graph).edges)
        degree_sequence = sorted((d for n, d in self._graph.degree()), reverse=True)
        dmax = max(degree_sequence)

        fig = plt.figure("Graph", figsize=(8, 8))
        # Create a gridspec for adding subplots of different sizes
        axgrid = fig.add_gridspec(1, 1)

        ax0 = fig.add_subplot(axgrid[0])
        pos = nx.spring_layout(self._graph)
        # pos = graphviz_layout(self._graph, prog="neato")
        nodes = list(self._graph.nodes())
        node_colors = [
            "black" if self._graph.nodes[n].get("message", False) else "gray"
            for n in nodes
        ]

        nx.draw_networkx_nodes(
            self._graph,
            pos,
            ax=ax0,
            nodelist=nodes,
            node_color=node_colors,
            node_size=30,
            margins=0.3,
        )
        nx.draw_networkx_edges(
            self._graph, pos, ax=ax0, alpha=0.4, node_size=30, width=2
        )
        ax0.set_axis_off()

        fig.tight_layout()
        plt.show(True)


g = Graph(c=0.2, p=0.5, delta_in=1, delta_out=1).initialize()

g.evolute(k=1000, rounds=5)
