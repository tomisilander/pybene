#!/usr/bin/env python

from argparse import ArgumentParser

import numpy as np
import torch

from .vd import fn2valcs
from .scorer import Scorer
from .benetypes import *
from .local_scores_gen import read_data, data2local_scores
from .local_scores_io import  read_local_scores, write_local_scores
from .constraints import file2musts_n_bans

def args2local_scores(args) -> LocalScores:
    valcounts = fn2valcs(args.vd_file)

    musts, bans = file2musts_n_bans(args.constraints) if args.constraints else ({},{})

    if args.dir:
        return read_local_scores(args.dir, len(valcounts))
    else:        
        data = read_data(args.data, valcounts)
        N = data.values().sum().item()
        scorer = Scorer(valcounts, N, args.score)
        return get_local_scores(valcounts, data, scorer, musts, bans)

def get_local_scores(data:torch.tensor, scorer:Scorer, 
                     musts={}, bans={}) -> LocalScores:
    return data2local_scores(data, scorer, musts=musts, bans=bans)

def negate(scores:LocalScores):
    """in place"""
    for ls in scores.values():
        for parset in ls:
            ls[parset] *= -1

def add_score_args(parser):
    parser.add_argument('-s', '--score', default='BIC')

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('vd_file')
    parser.add_argument('data')
    parser.add_argument('-o', '--outdir')
    parser.add_argument('-v', '--verbose', action='store_true')
    add_score_args(parser)
    args = parser.parse_args()
    args.dir = None

    local_scores = args2local_scores(args)
    if args.outdir:
        write_local_scores(local_scores, args.outdir)
    if args.verbose:
        print(local_scores)