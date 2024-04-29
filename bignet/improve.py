import numpy as np
import numpy.typing as npt
import networkx as nx
from itertools import chain
from argparse import ArgumentParser

from ..local_scores import add_score_args, get_local_scores
from ..scorer import Scorer
from ..best_net import add_args, save_net, best_net_in_S, negate
from ..best_subnet import project_by_vars, reindex_dict_of_sets, reindex_set
from ..vd import fn2valcs
from ..constraints import file2musts_n_bans
from ..beneDP import BeneDP

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

class Improver():

    def __init__(self, valcounts, data, musts, bans, rng):
        # type annotate later 
        self.valcounts = valcounts
        self.data = data
        self.musts = musts 
        self.bans = bans
        self.rng = rng
        
           
    def improve(self, g):
        # cut a piece and identify free nodes that can get new parents from within piece

        piece = cut_ancestors(g, 10, self.rng)
        piece_nodes = set(piece.nodes)
        free_nodes = {n for n in piece.nodes if set(g.predecessors(n)) <= piece_nodes}
        fixed_nodes = piece_nodes - free_nodes

        # print(piece_nodes)
        # print(free_nodes)
        # print(fixed_nodes)

        # optimice piece

        musts = {n:set(p for p in g.predecessors(n) if p in piece_nodes) 
                       for n in fixed_nodes}
        bans = {n:free_nodes for n in fixed_nodes}

        spiece_nodes = sorted(piece_nodes)
        valcounts,data,musts,bans = project_by_vars(self.valcounts, self.data, 
                                                    musts, bans, spiece_nodes)
        self.scorer.set_valcounts(valcounts)
        local_scores = get_local_scores(valcounts, data, self.scorer, musts, bans)
        
        # if args.worst:
            # negate(local_scores)
        bDP = BeneDP(local_scores)
        S = bDP.all_vars 
        best_net = best_net_in_S(S, bDP)
        
        # One should then reglue the piece back to the net
        ix2var = dict(enumerate(spiece_nodes))
        best_net = reindex_dict_of_sets(best_net, ix2var, spiece_nodes)
        
        to_be_removed = [(p,n) for n in free_nodes for p in g.predecessors(n)]
        to_be_added =   [(p,n) for n in free_nodes for p in best_net[n]]
        
        g.remove_edges_from(to_be_removed)
        g.add_edges_from(to_be_added)
        
        
def get_random_graph(n, p, rng):
    return get_random_dag(n ,p, rng)

    
def args_2_big_net(args):

    valcounts = fn2valcs(args.vd_file)

    # reindex musts and bans
    musts, bans = file2musts_n_bans(args.constraints) if args.constraints else ({},{})

    # just read relevant data - hmm, could be already in memory    
    data_mx = np.loadtxt(args.data_file, valcounts)
    N = data_mx.shape[0]
    
    scorer = Scorer(valcounts, N, args.score)

    return None
  

def add_args(parser:ArgumentParser):
    parser.add_argument('vd_file')
    add_score_args(parser)


if __name__ == '__main__':
    parser = ArgumentParser()
    add_args(parser)
    parser.add('--seed', type=float)
    args = parser.parse_args()
    best_net, best_score = args_2_big_net(args)
    
    print(best_net, best_score)
    if args.outfile:
        save_net(best_net, args.outfile)