#!/usr/bin/env python

from argparse import ArgumentParser

import numpy as np

from vd import fn2valcs
from scorer import Scorer
from benetypes import *
from local_scores_gen import read_data, data2local_scores
from local_scores_io import  read_local_scores, write_local_scores

def args2local_scores(args):
    valcounts = fn2valcs(args.vd_file)
    if args.dir:
        return read_local_scores(args.dir, len(valcounts))
    else:        
        # if there are vars could one then just read those
        data = read_data(args.data, valcounts)
        N = data.values().sum().item()
        scorer = Scorer(valcounts, N, args.score)
        return data2local_scores(valcounts, data, scorer)

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