import networkx as nx
from scipy.special import gammaln
from random import choices, seed, randint
import numpy as np
from gnuplot import export_for_gnuplot
import itertools
import matplotlib.pyplot as plt

def get_probabilities(n: int, c: float):
    # Пусть у нас 1 <= k <=100
    def poisson(n: int, k: int, c: float):
        def slow_var_f(k: int):
            return np.log(8 * k)

        def lambda_f(k: int, c: float):
            return np.pow(k, c) * slow_var_f(k)

        return (
            np.exp(-1 * lambda_f(n + 1, c))
            * (np.pow(lambda_f(n + 1, c), k - 1))
            / gammaln(k)
        )

    return np.array([poisson(n, k, c) for k in range(1, K_SIZE + 1)])


def get_log_probabilities(n: int, c: float):
    def lambda_f(k, c):
        return np.power(k, c) * np.log(8 * k)

    l = lambda_f(n + 1, c)
    ks = np.arange(1, K_SIZE + 1)

    log_probs = -l + (ks + 1) * np.log(l) - gammaln(ks)

    weights = np.exp(log_probs - np.max(log_probs))
    return weights


# def probability_in_node(G: nx.MultiDiGraph, node: int, delta_in: float):
#     return (G.in_degree(node) + delta_in) / (
#         sum([t[1] for t in G.in_degree()]) + G.number_of_nodes() * delta_in
#     )


# def probability_out_node(G: nx.MultiDiGraph, node: int, delta_out: float):
#     return (G.out_degree(node) + delta_out) / (
#         sum([t[1] for t in G.out_degree()]) + G.number_of_nodes() * delta_out
#     )

seed(randint(10**2, 10**7))
G = nx.MultiDiGraph()
G.add_node(0, message=True)
G.add_edge(0, 0)

c = 0.45
p = 0.8
delta_in = delta_out = 2
K_SIZE = 10000 
rounds = 5
for idx, _ in enumerate(range(rounds)):
    K = get_log_probabilities(G.number_of_nodes(), c=c)
    M = choices(population=list(range(1, K_SIZE + 1)), weights=K.tolist())[0]
    print(f"M = {M}")
    probabilities_in = np.array(
        [(G.in_degree(n) + delta_in) for n in range(G.number_of_nodes())]
    ).dot(1 / (sum(t[1] for t in G.in_degree()) + G.number_of_nodes() * delta_in))
    probabilities_out = np.array(
        [(G.out_degree(n) + delta_out) for n in range(G.number_of_nodes())]
    ).dot(1 / (sum(t[1] for t in G.out_degree()) + G.number_of_nodes() * delta_out))
    graph_nodes = set(G.nodes)
    nodes = list(itertools.product(graph_nodes, graph_nodes))
    probabilities_final = np.multiply(
        [probabilities_in[n1] for n1, _ in nodes],
        [probabilities_out[n2] for _, n2 in nodes],
    )
    last_node = G.number_of_nodes() - 1
    edges = []
    node_changes = {}
    node_pointer = last_node
    for i in range(M):
        if choices(population=(0, 1), weights=(p, 1 - p))[0] == 0:
            node_pointer += 1

            node_2 = choices(
                population=list(range(G.number_of_nodes())),
                weights=probabilities_in.tolist(),
            )[0]

            edges.append((node_pointer, node_2))
        else:
            edge = choices(
                population=list(nodes),
                weights=probabilities_final.tolist(),
            )[0]
            edges.append(edge)
            from_node, to_node = G.nodes[edge[0]], G.nodes[edge[1]]

            if from_node.get("message"):
                node_changes[edge[1]] = {"message": True}
    G.add_edges_from(edges)
    for key, value in node_changes.items():
        node = G.nodes[key]
        node.update(value)
    node_changes = {}
    edges = []
print(G)

# export_for_gnuplot(G)

def draw_graph(G):
    degree_sequence = sorted((d for n, d in G.degree()), reverse=True)
    dmax = max(degree_sequence)

    fig = plt.figure("Degree of a random graph", figsize=(8, 8))
    # Create a gridspec for adding subplots of different sizes
    axgrid = fig.add_gridspec(5, 4)

    ax0 = fig.add_subplot(axgrid[0:3, :])
    pos = nx.spring_layout(G)
    nx.draw_networkx_nodes(G, pos, ax=ax0, node_size=10, margins=0)
    nx.draw_networkx_edges(G, pos, ax=ax0, alpha=0.4)
    ax0.set_title("Connected components of G")
    ax0.set_axis_off()

    ax1 = fig.add_subplot(axgrid[3:, :2])
    ax1.plot(degree_sequence, "b-", marker="o")
    ax1.set_title("Degree Rank Plot")
    ax1.set_ylabel("Degree")
    ax1.set_xlabel("Rank")

    ax2 = fig.add_subplot(axgrid[3:, 2:])
    ax2.bar(*np.unique(degree_sequence, return_counts=True))
    ax2.set_title("Degree histogram")
    ax2.set_xlabel("Degree")
    ax2.set_ylabel("# of Nodes")

    fig.tight_layout()
    plt.show(True)

draw_graph(G)