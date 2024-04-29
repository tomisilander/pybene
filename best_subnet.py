#!/usr/bin/env python

from typing import Iterator
from argparse import ArgumentParser
import pathlib
import pickle

import numpy as np

from benetypes import *
from beneDP import BeneDP
from vd import fn2valcs
from local_scores import add_score_args, negate, Scorer, get_local_scores, file2musts_n_bans
from local_scores_gen import data_mx_to_coo
from best_net import best_net_in_S

def reindex_set(s, rixer):
    return set(map(rixer.get, s))

def reindex_dict_of_sets(d:dict[int,set[int]], rixer:dict[int,int], keys:Iterator[int]):
    for key in keys:
        if key in d:
            yield (rixer[key], reindex_set(d[key], rixer))

def project_by_vars(valcounts, data, musts, bans, vars):
    valcounts = [valcounts[v] for v in vars]
    data = data[:,vars]

    var2ix = {v:i for (i,v) in enumerate(vars)}
    musts = dict(reindex_dict_of_sets(musts, var2ix, vars))
    bans = dict(reindex_dict_of_sets(bans, var2ix, vars))
    
def args2local_scores(args, vars:Iterator[int]) -> LocalScores:

    vars = sorted(vars)
    valcounts = fn2valcs(args.vd_file)
    valcounts = [valcounts[v] for v in vars]

    # reindex musts and bans
    var2ix = {v:i for (i,v) in enumerate(vars)}
    musts, bans = file2musts_n_bans(args.constraints) if args.constraints else ({},{})
    musts = dict(reindex_dict_of_sets(musts, var2ix, vars))
    bans = dict(reindex_dict_of_sets(bans, var2ix, vars))

    # just read relevant data - hmm, could be already in memory    
    data = data_mx_to_coo(np.loadtxt(args.data_file, usecols=vars), valcounts)
    N = data.values().sum().item()
    
    scorer = Scorer(valcounts, N, args.score)
    print('VC', valcounts, scorer.valcounts)
    ls =  get_local_scores(valcounts, data, scorer, musts, bans)
    return ls

def add_args(parser:ArgumentParser):
    parser.add_argument('vd_file')
    parser.add_argument('data_file')
    add_score_args(parser)
    parser.add_argument('--constraints')
    parser.add_argument('--worst', action='store_true')
    parser.add_argument('--vars', nargs='+', type=int, required=True)
    parser.add_argument('-o', '--outfile')

def save_net(net:Net, outfile):
    outpath = pathlib.Path(outfile)
    outdir = outpath.parent
    outdir.mkdir(parents=True, exist_ok=True)
    pickle.dump(net, outpath.open('wb'))

if __name__ == '__main__':
    parser = ArgumentParser()
    add_args(parser)
    args = parser.parse_args()
    
    local_scores = args2local_scores(args, args.vars)
    if args.worst:
        negate(local_scores)
    bDP = BeneDP(local_scores)
    S = bDP.all_vars 
    best_net = best_net_in_S(S, bDP)
    ix2var = dict(enumerate(args.vars))
    best_net = dict(reindex_dict_of_sets(best_net, ix2var, best_net))
    print(best_net, bDP.best_net_score_in[S])
    if args.outfile:
        save_net(best_net, args.outfile)