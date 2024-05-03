import numpy as np
import numpy.typing as npt
import networkx as nx
from itertools import chain
from argparse import ArgumentParser
from collections import defaultdict

from ..local_scores import add_score_args, get_local_scores
from ..local_scores_gen import data_mx_to_coo, contab2condtab
from ..scorer import Scorer
from ..best_net import add_args, save_net, best_net_in_S, negate
from ..best_subnet import project_by_vars, reindex_dict_of_sets, reindex_set
from ..vd import fn2valcs
from ..constraints import file2musts_n_bans, parentize
from ..beneDP import BeneDP
from ..common import load_bn

def get_score(g, data_mx, v, scorer):
    valcounts = scorer.valcounts
    assert data_mx.shape[1] == len(valcounts)
    ps = sorted(g.predecessors(v))
    vps = [v] + ps
    vps_mx = data_mx[:, vps]
    contab = data_mx_to_coo(vps_mx, [valcounts[vp] for vp in vps])
    condtab = contab2condtab(contab, 0, valcounts[v])
    # print('getting score',v,ps,condtab)
    score = scorer.score(v, ps, condtab)
    return v, ps, score

def gen_scores(g, data_mx, vars, scorer):
    for v in vars:
        yield get_score(g, data_mx, v, scorer)

def score_net(g, data_mx, scorer):
    for v in g.nodes:
        yield get_score(g, data_mx, v, scorer)
      
def get_random_dag(n:int, p:float, rng):
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

def is_cyclic(g:nx.DiGraph) -> bool:
    try: 
        return nx.find_cycle(g, orientation='original')
    except nx.NetworkXNoCycle:
        return False

def gen_banned_decendants(g:nx.DiGraph, s:set):
    for n in s:
        decs = nx.descendants(g, n)
        targets = s & decs
        vias = {t:set() for t in targets}
        for path in nx.all_simple_paths(g,n,targets):
            vias[path[-1]] |= set(path)
        for d, viaset in vias.items():
            if not viaset <= s:
                yield (d,n)

class Improver():

    def __init__(self, g, data_mx, musts, bans, scorer, rng):
        # type annotate later
        self.g = g 
        self.orig_valcounts = scorer.valcounts
        self.data_mx = data_mx
        self.musts = musts 
        self.bans = bans
        self.scorer = scorer
        self.rng = rng
        self.score_table = self.get_score_table(g)
        
    def get_score_table(self, g):
        score_table = [((),0.0)] * len(self.scorer.valcounts)
        for v,ps, s in score_net(g, self.data_mx, self.scorer):
            score_table[v] = (ps, s)
        return score_table
    
    def improve(self):
        """Takes a piece of graph, optimizes it, and glues it back. 
           This should never make the results worse."""

        g = self.g

        # CUT a piece and identify free nodes that can get new parents from within piece
        
        piece = cut_ancestors(g, 10, self.rng)
        piece_nodes = set(piece.nodes)
        free_nodes = {n for n in piece.nodes if set(g.predecessors(n)) <= piece_nodes}
        fixed_nodes = piece_nodes - free_nodes # should keep their parents 
        
        musts = {n : (set(g.predecessors(n)) | self.musts[n]) & piece_nodes 
                       for n in fixed_nodes} 
        bans  = {n : piece_nodes - set(g.predecessors(n))   
                      for n in fixed_nodes}

        # HEY! should also include old musts and bans

        spiece_nodes = sorted(piece_nodes)
        
        new_bans = parentize(gen_banned_decendants(g,piece_nodes))
        for n, nbs in new_bans.items():
            bans[n] = bans.get(n,set()) | nbs
            
        print(spiece_nodes)
        # report the scores of the free nodes in a piece 
        print('fixed nodes:')
        for n in sorted(fixed_nodes):
            p,s = self.score_table[n]
            print(n, p, s)
        print('free nodes:')
        tot_before = 0.0
        for n in sorted(free_nodes):
            p,s = self.score_table[n]
            print(n, p, s)
            tot_before += s
        print('Tot: ',tot_before)
        print()

        # OPTIMIZE PIECE

        # project to new indices
        prj_valcounts, prj_data_mx, prj_musts, prj_bans \
            = project_by_vars(self.scorer.valcounts, self.data_mx, musts, bans, spiece_nodes)
        
        prj_data_coo = data_mx_to_coo(prj_data_mx, prj_valcounts)

        # new scorer too
        piece_scorer = Scorer(prj_valcounts, prj_data_mx.shape[0], 
                              self.scorer.score_name, **self.scorer.kwargs)

        prj_local_scores = get_local_scores(prj_data_coo, piece_scorer, prj_musts, prj_bans)
        # if args.worst:
            # negate(local_scores)
        bDP = BeneDP(prj_local_scores)
        best_net = best_net_in_S(bDP.all_vars, bDP)


        # report the scores of the free nodes in an optimized piece 
        ix2var = dict(enumerate(spiece_nodes))
        best_edges = ((p,c) for c,ps in best_net.items() for p in ps)
        best_g = nx.DiGraph()
        best_g.add_edges_from(best_edges)

        print('fixed nodes:')
        for n in sorted(fixed_nodes):
            p,s = self.score_table[n]
            print(n, p, s)
        print('free nodes after:')
        tot_after = 0.0
        for n,ps,s in sorted(score_net(best_g, prj_data_mx, piece_scorer)):
            v = ix2var[n]
            if v in free_nodes:
                print(v,sorted(tuple(map(ix2var.get,ps))),s)
                tot_after += s
        print('Tot:', tot_after, 'Improvement: ', tot_after-tot_before)
        print() 
        
                       
                       
        # REPLACE THE PIECE WITH AN OPTIMIZED ONE
        best_net = dict(reindex_dict_of_sets(best_net, ix2var, range(len(spiece_nodes))))
        to_be_removed = [(p,n) for n in free_nodes for p in g.predecessors(n)]
        to_be_added =   [(p,n) for n in free_nodes for p in best_net.get(n,())]
        
        g.remove_edges_from(to_be_removed)
        g.add_edges_from(to_be_added)
        
def get_random_graph(n, p, rng):
    return get_random_dag(n ,p, rng)

    
def args_2_big_net(args):
    valcounts = fn2valcs(args.vd_file)
    data_mx = np.loadtxt(args.data_file)
    N,n = data_mx.shape

    g = load_bn(args.bn_file) if args.bn_file else nx.empty_graph(n, create_using=nx.DiGraph)

    musts, bans = defaultdict(set), defaultdict(set)
    if args.constraints:
        musts, bans = file2musts_n_bans(args.constraints)
    
    scorer = Scorer(valcounts, N, args.score)
    
    rng = np.random.default_rng(args.seed)

    improver = Improver(g, data_mx, musts, bans, scorer, rng)
    print('start', sum(s for (_p,s) in improver.score_table))

    for r in range(50):
        improver.improve()
        improver.score_table = improver.get_score_table(improver.g)        
        total_score = sum(s for (_p,s) in improver.score_table)
        print(total_score)
        ic =  is_cyclic(improver.g)
        if ic:
            print(ic)
            assert not ic
    return g, total_score
  

def add_args(parser:ArgumentParser):
    parser.add_argument('vd_file')
    parser.add_argument('--data-file')
    parser.add_argument('--bn-file')
    parser.add_argument('--constraints')
    parser.add_argument('--outfile')
    parser.add_argument('--dotfile')
    
    parser.add_argument('--seed', type=int)
    add_score_args(parser)


if __name__ == '__main__':
    parser = ArgumentParser()
    add_args(parser)
    args = parser.parse_args()
    best_net, best_score = args_2_big_net(args)
    
    print(best_net, best_score)
    if args.outfile:
        save_net(best_net, args.outfile)
    if args.dotfile:
        nx.nx_agraph.write_dot(best_net, args.dotfile)