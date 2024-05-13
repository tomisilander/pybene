from itertools import product, combinations, count
import numpy as np
import networkx as nx
from argparse import ArgumentParser

from ..common import load_bn

def dist_iter(g:nx.DiGraph):
    ug = g.to_undirected(as_view=True)
    dist_iter = nx.all_pairs_shortest_path_length(ug) 
    return dist_iter

def get_dst_mx(dst_dict:dict):
    n = len(dst_dict)
    return np.fromiter((dst_dict[src][dst] for src,dst in product(range(n), repeat=2)), 
                       dtype=float, count=n*n).reshape(n,n)

def gen_distant_nodes(dst_mx, n=None, mindist=None):
    assert not (n is None and mindist is None)
    
    nodes = list(np.divmod(np.argmax((dst_mx)), dst_mx.shape[0]))
    d =dst_mx[nodes[0],nodes[1]]
    for i in count():
        if n and i>n: break
        if mindist and d < mindist: break
        min_dsts = dst_mx[nodes,:].min(axis=0)
        new_node = np.argmax(min_dsts)
        d = min_dsts[new_node]
        nodes.append(new_node)

    return nodes

def print_dsts(dst_mx, nodes):
    for src,dst in combinations(nodes, 2):
        print(src, dst, dst_mx[src,dst])


def add_args(parser:ArgumentParser):
    parser.add_argument('bn_file')    
    parser.add_argument('-n', '--number', type=int)    
    parser.add_argument('-d','--mindist', type=int)    
    parser.add_argument('--seed', type=int)


if __name__ == '__main__':
    parser = ArgumentParser()
    add_args(parser)
    args = parser.parse_args()
    
    g = load_bn(args.bn_file)
    diter = dist_iter(g)
    dsts = get_dst_mx(dict(diter))
    nodes = gen_distant_nodes(dsts, args.number, args.mindist)
    print_dsts(dsts, nodes)