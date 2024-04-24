import numpy as np
import networkx as nx
from itertools import chain

def get_random_dag(n, p, rng):
    mx = rng.choice(2, p=(1-p, p), size=(n,n))
    mx = np.tril(mx, k=-1)
    return nx.from_numpy_array(mx, create_using=nx.DiGraph)
    
def get_free_nodes(g, nodes):
    return {n for n in nodes if set(g.predecessors(n)) <= nodes}
    
def cut_ancestors(g: nx.Graph, nof_nodes:int, rng:np.random.Generator) -> set:
    # return such a subgraph sg, consisting of ancestors of a random node n, that
    # the subgraph has over nof_nodes of "free" nodes whose all parents are in sg
    # but can I now optimize free nodes - no - searching for a correct cut
    g = g.copy()
    nset, new_parents = set(), set()
    n = rng.choice(g.nodes)
    print('start', n)
    nset.add(n)
    new_parents.add(n) 
    while len(nset) < nof_nodes:
        new_parents = set(chain(*map(g.predecessors, new_parents))) - nset
        if not new_parents:
            print('OOP')
            break
        nset.update(new_parents) # should I select randomly
            
    return nx.subgraph(g, nset)

rng = np.random.default_rng()
g = get_random_dag(40 ,0.2, rng=rng
                   )
piece = cut_ancestors(g, 10, rng)

piece_nodes = set(piece.nodes)
free_nodes = {n for n in piece.nodes if set(g.predecessors(n)) <= piece_nodes}
fixed_nodes = piece_nodes - free_nodes
print(piece_nodes)
print(free_nodes)
print(fixed_nodes)

# one should now optimise piece_nodes keeping edges of piece(!) to fixed nodes fixed 
# One should then reglue the piece back to the net
# One could also learn to cut pieces of correct size
    
          