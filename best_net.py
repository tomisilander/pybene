#!/usr/bin/env python

from typing import Iterator, Sequence
from argparse import ArgumentParser
import pathlib
import pickle

from benetypes import *
from beneDP import BeneDP
from vd import fn2valcs
from local_scores import add_score_args, args2local_scores, negate

def best_net_in_S(S : Varset, bDP : BeneDP) -> Net:

    def gen_best_order_in_S(S : Varset)  -> Iterator[Var]:

        def best_sink_in_S(S : Varset) -> Var:

            def gen_sink_scores():
                for x in S:
                    S_x = S - {x}
                    parents = bDP.best_parents_4in[x][S_x]
                    parents_score = bDP.local_scores[x][parents]
                    score = bDP.best_net_score_in[S_x] + parents_score
                    yield  (score, x)

            return max(gen_sink_scores())[1]

        if len(S)==0 : return
        s = best_sink_in_S(S)
        yield from gen_best_order_in_S(S-{s})
        yield s

    def best_net_for_order(S : Varset, order : Sequence[Var]) -> Net :
        if len(order) == 0:
            return dict()
        sink = order[-1]
        S_sink = S - {sink}
        best_net = best_net_for_order(S_sink, order[:-1])
        best_net.update({sink : bDP.best_parents_4in[sink][S_sink]})
        return  best_net

    order = list(gen_best_order_in_S(S))
    return best_net_for_order(S, order)

def add_args(parser:ArgumentParser):
    parser.add_argument('vd_file')
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument('--data')
    g.add_argument('--dir')
    add_score_args(parser)
    parser.add_argument('--constraints')
    parser.add_argument('--worst', action='store_true')
    parser.add_argument('--vars', nargs='+', type=int)
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
    local_scores = args2local_scores(args)
    if args.worst:
        negate(local_scores)
    bDP = BeneDP(local_scores)
    S = frozenset(args.vars) if args.vars else bDP.all_vars 
    best_net = best_net_in_S(S, bDP)
    print(best_net, bDP.best_net_score_in[S])
    if args.outfile:
        save_net(best_net, args.outfile)