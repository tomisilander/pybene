#!/usr/bin/env python

import networkx as nx

from local_scores import args2local_scores
from benetypes import *

def gen_weighted_arcs(ls:LocalScores, S:Varset):
    emptyset = frozenset()
    singletons = dict((x, frozenset([x])) for x in S)
    for x in S:
        lsx = ls[x]
        score0 = lsx[emptyset]
        for p in S:
            if p == x: continue
            pscore = lsx[singletons[p]]
            if True or pscore > score0 :
                yield (p, x, {'weight' : pscore})

def find_bad_edges(g:nx.DiGraph, ls:LocalScores) -> list:
    emptyset = frozenset()
    return [(p,c) for p, c, weight in g.edges.data('weight') 
            if weight<ls[c][emptyset]]

def get_best_forest(ls:LocalScores, S:Varset) -> nx.DiGraph:
    g = nx.DiGraph()
    g.add_edges_from(gen_weighted_arcs(ls, S))
    b = nx.algorithms.tree.branchings.maximum_spanning_arborescence(g)
    bad_edges = find_bad_edges(b, ls)
    b.remove_edges_from(bad_edges)
    return b

def score_forest(b:nx.DiGraph, ls:LocalScores) -> Score:
    emptyset = frozenset()
    return sum(ls[n][emptyset] if b.in_degree(n) == 0 else 
               ls[n][frozenset([next(b.predecessors(n))])]
               for n in b.nodes())

if __name__ == '__main__':
    from argparse import ArgumentParser
    from best_net import add_args

    parser = ArgumentParser()
    add_args(parser)
    args = parser.parse_args()

    local_scores = args2local_scores(args)
    S = frozenset(args.vars if args.vars else local_scores.keys())
    b = get_best_forest(local_scores,S)
    print(score_forest(b, local_scores), b.edges())