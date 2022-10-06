#!/usr/bin/env python

from collections import defaultdict

import numpy as np
import torch
from torch.sparse import sum as marginalize

from scorer import Scorer
from benetypes import *

def read_data(filename:str, valcounts:np.ndarray) -> torch.tensor:
    """reads data to a (sparse) torch tensor"""
    i = np.loadtxt(filename, dtype=int).transpose()
    v = torch.ones(i.shape[1], dtype=int)
    return torch.sparse_coo_tensor(i,v,tuple(valcounts)).coalesce()

def gen_sets_down(bitset, first_out_ix):
    """recursively find which variable should be marginalized out and from where"""
    var_count = sum(map(int, bin(bitset)[2:]))
    if var_count <= 1:
        return
    for x in range(first_out_ix):
        yield (var_count, x)
        xset = 1<<x
        next_set = bitset ^ xset # remove x
        yield from gen_sets_down(next_set, x)

def gen_contabs(start_contab):
    """sequantially marginalize contabs as dictated by gen_sets_down"""
    n = start_contab.sparse_dim()
    contabs = [None]*(n+1)
    contabs[n] = (tuple(range(n)), start_contab)
    yield contabs[n]
    start_bitset = (1<<n)-1
    for var_count, x in gen_sets_down(start_bitset, n):
        old_vars, old_contab = contabs[var_count]
        pos_x = old_vars.index(x)
        new_vars = old_vars[:pos_x]+old_vars[pos_x+1:]
        new_contab = marginalize(old_contab,[pos_x])
        contabs[var_count-1] = (new_vars, new_contab)
        yield contabs[var_count-1]

def contab2condtab(contab, i, valcount):
    if contab.sparse_dim() == 1:
        pcfg2ix = {():0}
        pcfg_count = 1
    else:
        pcfgs = marginalize(contab, [i]).indices().numpy().transpose()
        pcfg_count = pcfgs.shape[0]
        pcfg2ix = dict(zip(map(tuple,pcfgs), range(pcfg_count)))

    cfgs = contab.indices().numpy().transpose()
    condtab = np.zeros((pcfg_count, valcount))
    for cfg, freq in zip(map(tuple,cfgs), contab.values().numpy()):
        pcfg = cfg[:i] + cfg[i+1:]
        pcfg_ix = pcfg2ix[pcfg]
        condtab[pcfg_ix][cfg[i]] = freq
    return condtab

def gen_condtabs(contabs, valcounts):
    for s, contab in contabs:
        for i,x in enumerate(s):
            condtab = contab2condtab(contab, i, valcounts[x])
            ps = s[:i]+s[i+1:]
            yield x, ps, condtab

def gen_local_scores(condtabs, scorer):
    for x, ps, condtab in condtabs:
        yield x, ps, scorer.score(x,ps,condtab)

def get_local_scores(local_scores_gen):
    local_scores = defaultdict(dict)
    for x, ps, score in local_scores_gen:
        local_scores[x][frozenset(ps)] = score
    return local_scores

def data2local_scores(valcounts, data, scorer):
    contabs = gen_contabs(data)
    condtabs = gen_condtabs(contabs, valcounts)
    local_scores = gen_local_scores(condtabs, scorer)
    return get_local_scores(local_scores)
