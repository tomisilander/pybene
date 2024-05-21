import pathlib
import networkx as nx

def save_bn(G, bn_filename):
    dirname = pathlib.Path(bn_filename).parent
    dirname.mkdir(parents=True, exist_ok=True)
    with open(bn_filename, 'w') as outf:
        print(G.number_of_nodes(), file=outf)
        for (x,y) in G.edges():
            print(x, y, file=outf)

def load_bn(bn_filename):
    with open(bn_filename) as inf:
        G = nx.DiGraph()
        G.add_nodes_from(range(int(inf.readline())))
        edges = [tuple(map(int, l.split())) for l in inf]        
        G.add_edges_from(edges)
        return G
