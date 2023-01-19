import numpy as np
import networkx as nx

def get_subset(g:nx.Graph, rng:np.random.Generator):
    MAX_N = 10
    n = g.number_of_nodes()
    if n<=MAX_N:
        return set(g.nodes)

    i = rng.integers(n)
    nset = {i}
    free_space = MAX_N - len(nset)
    while free_space > 0:
        bs = list(nx.node_boundary(g,nset))
        nset |= set(rng.choice(bs, min(free_space, len(bs)), replace=False))
        free_space = MAX_N - len(nset)
    return nset

# to test getting a subset

rng = np.random.default_rng()
g = nx.gnp_random_graph(20,0.2, directed=True)
s = get_subset(g.to_undirected(), rng)
print(s)
print(g.edges)