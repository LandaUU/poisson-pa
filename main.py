import networkx as nx
import math
import scipy.special as special
from random import choices
import numpy as np

def get_probabilities(n: int, c: float):
    # Пусть у нас 1 <= k <=100
    def poisson(n: int, k: int, c: float):
        def slow_var_f(k: int):
            return np.log(8*k)
        def lambda_f(k: int, c: float):
            return np.pow(k,  c) * slow_var_f(k)
        return np.exp(-1 * lambda_f(n+1, c)) * (np.pow(lambda_f(n+1, c), k-1)) / special.factorial(k-1)

    return np.array([poisson(n, k, c) for k in range(1, 501)])

def probability_in_node(G: nx.MultiDiGraph, node: int, delta_in: float):
    return (G.in_degree(node) + delta_in)/ (sum([t[1] for t in G.in_degree()]) + G.number_of_nodes() * delta_in)

def probability_out_node(G: nx.MultiDiGraph, node: int, delta_out: float):
    return (G.out_degree(node) + delta_out)/ (sum([t[1] for t in G.out_degree()]) + G.number_of_nodes() * delta_out)

G = nx.MultiDiGraph()
G.add_node(0)
G.add_edge(0, 0)

c = 0.2
p = 0.5

rounds = 3

for _ in range(rounds):

    K = get_probabilities(G.number_of_nodes(), c=c)
    M = choices(population=list(range(1, 201)), weights=K.tolist())[0]

    for i in range(M):
        if choices(population=(0, 1), weights=(p, 1-p))[0] == 0:
            G.add_node(G.number_of_nodes())
            node_1 = G.number_of_nodes()

            node_2 = choices(population=list(range(G.number_of_nodes())), weights=[probability_in_node(G, n, 2) for n in range(G.number_of_nodes())])[0]

            G.add_edge(node_1, node_2)
        else:
            node_1 = choices(population=list(range(G.number_of_nodes())), weights=[probability_in_node(G, n, 2) for n in range(G.number_of_nodes())])[0]
            s2 = list(range(G.number_of_nodes())) 
            node_2 = choices(population=s2, weights=[probability_out_node(G, n, 2) for n in s2])[0]

            G.add_edge(node_1, node_2)

print(G)