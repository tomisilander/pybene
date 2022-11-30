#!/usr/bin/env python

import numpy as np
from numpy.random import default_rng
import torch
import ray

from vd import fn2valcs
from best_net import add_score_args, best_net_in_S
from beneDP import BeneDP
from local_scores_gen import data2local_scores
from scorer import Scorer

@ray.remote
class Bene:
    def __init__(self, vd_file, data_file, seed=None):
        self.valcounts = np.array(fn2valcs(vd_file), dtype=int)
        self.datamxT = np.loadtxt(data_file, dtype=int).transpose()
        self.N = self.datamxT.shape[1]
        self.rng = default_rng(seed)

    def get_shape(self):
        nof_vars, N = self.datamxT.shape
        return N, nof_vars

    def slice_data(self, S, N):
        selvars = np.array(sorted(S))
        selvalcounts = self.valcounts[selvars]
        if N is None:
            N = self.N
            seldata = self.datamxT[selvars,:]
        else:
            ixs = self.rng(self.N, N, replace=False, shuffle=False)
            seldata = self.datamxT[selvars, ixs]
        
        return selvars, selvalcounts, seldata, N

    def get_best_in_S(self, S, score, N=None):

        selvars, selvalcounts, seldata, N = self.slice_data(S, N)

        v = torch.ones(N, dtype=int)
        datensor = torch.sparse_coo_tensor(seldata,v,tuple(selvalcounts)).coalesce()
        scorer = Scorer(selvalcounts, N, score)

        local_scores = data2local_scores(selvalcounts, datensor, scorer)
        bDP = BeneDP(local_scores)    

        S = frozenset(range(len(selvars)))
        best_net = best_net_in_S(S, bDP)

        return bDP.best_net_score_in[S], best_net


if __name__ == '__main__':
    from argparse import ArgumentParser
    
    parser = ArgumentParser()
    parser.add_argument('vd_file')
    parser.add_argument('data_file')
    parser.add_argument('--vars', nargs='+', type=int)
    parser.add_argument('-N', type=int)
    add_score_args(parser)
    args = parser.parse_args()

    bene = Bene.remote(args.vd_file, args.data_file)
    shapefuture = bene.get_shape.remote()
    _N, nof_vars = ray.get(shapefuture)

    S = frozenset(args.vars if args.vars else range(nof_vars))
    futures = bene.get_best_in_S.remote(S, args.score, args.N)
    print(ray.get(futures))
