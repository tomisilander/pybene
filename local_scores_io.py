#!/usr/bin/env python

from pathlib import Path
from array import array
from typing import Iterator
from benetypes import *

def int2set(x:int) -> Varset:

    def gen_memebers(x):
        i = 0
        while x > 0:
            if x & 1:
                yield i
            i += 1
            x >>= 1

    return frozenset(gen_memebers(x))

def set2int(s:Varset) -> int:
    return sum(1<<x for x in s)

def parset2varset(S:Varset, x:Var) -> Varset:
    return frozenset(s if s<x else s+1 for s in S)

def varset2parset(S:Varset, x:Var):
    return frozenset(s if s<x else s-1 for s in S)

def gen_parvarsets(n:int, x:Var) -> Iterator[Varset]:
    for i in range(2**(n-1)):
        yield parset2varset(int2set(i), x)

def read_local_scores(dirname, n) -> LocalScores:
    resdir = Path(dirname)
    local_scores = dict()
    for x in range(n):
        with open(resdir/str(x), 'rb') as f:
            a = array('d')
            a.fromfile(f, 2**(n-1))
            local_scores[x] = dict(zip(gen_parvarsets(n, x), a))

    return local_scores

def write_local_scores(local_scores:LocalScore, dirname:str):
    resdir = Path(dirname)
    resdir.mkdir(exist_ok=True, parents=True)
    n = len(local_scores)
    for x in range(n):
        with open(resdir/str(x), 'wb') as f:
            sorted_score_items = sorted(local_scores[x].items(), key=lambda i: set2int(i[0]))
            scores = list(zip(*sorted_score_items))[1]
            a = array('d', scores)
            a.tofile(f)

    return local_scores
