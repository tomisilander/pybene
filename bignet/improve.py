import numpy as np
import networkx as nx

def get_subset(g:nx.Graph, max_nof_nodes:int, rng:np.random.Generator) -> set:
    """_select a random node and nodes around it_

    Args:
        g (nx.Graph): _a graph from which a subset will be selected_
        max_nof_nodes (int): _maximum number of nodes selected_
        rng (np.random.Generator): _random number generator_

    Returns:
        _set_: _set of nodes connected in g_
    """
    n = g.number_of_nodes()
    if n<=max_nof_nodes:
        return set(g.nodes)

    i = rng.integers(n)
    nset = {i}
    free_space = max_nof_nodes - len(nset)
    while free_space > 0:
        bs = list(nx.node_boundary(g, nset))
        nset |= set(rng.choice(bs, min(free_space, len(bs)), replace=False))
        free_space = max_nof_nodes - len(nset)
    return nset

def cut_piece(g: nx.Graph, nof_nodes:int, rng:np.random.Generator) -> set:
    ug = g.to_undirected()
    nset = set()
    while len(nset) < nof_nodes and ug.number_of_nodes() > 0:
        nodes_left = nof_nodes - len(nset)
        s = get_subset(ug, nodes_left, rng)
        nset |= s
        ug.remove_nodes_from(s)
    return nx.subgraph(g, nset)


rng = np.random.default_rng()
g = nx.gnp_random_graph(20,0.2, directed=True)
piece = cut_piece(g, 30, rng)
print(piece.edges)
