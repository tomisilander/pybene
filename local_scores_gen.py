#!/usr/bin/env python

from collections import defaultdict

import numpy as np
import torch
from torch.sparse import sum as marginalize

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
    """sequentially marginalize contabs as dictated by gen_sets_down"""
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
    cfgs  = contab.indices().numpy().transpose()
    i_vals = cfgs[:,i]
    freqs = contab.values().numpy()
    pcfgs = cfgs.copy()
    pcfgs[:,i] = -1

    bcfgs = pcfgs.tobytes()
    bcfg_len = cfgs.shape[1] * 8    
    starts = range(0, 8*cfgs.size, bcfg_len)
    ends   = range(bcfg_len,8*cfgs.size+1,bcfg_len)
    
    condtab = np.zeros((cfgs.shape[0], valcount), dtype=np.int64) # too big but
    pcfg2ix = {}
    # slow loop but
    for (start,end,i_val,freq) in zip(starts,ends,i_vals,freqs):
        pcfg_ix = pcfg2ix.setdefault(bcfgs[start:end], len(pcfg2ix))        
        condtab[pcfg_ix][i_val] = freq
    return condtab[:len(pcfg2ix),:]

def gen_condtabs(contabs, valcounts):
    for s, contab in contabs:
        for i, x, in enumerate(s):
            condtab = contab2condtab(contab, i, valcounts[x])
            ps = s[:i]+s[i+1:]
            yield x, ps, condtab

def gen_local_scores(condtabs, scorer, must_parents={}, banned_parents={}):
    empty =set()
    for x, ps, condtab in condtabs:
        pset = set(ps)
        has_banned_parents = len(banned_parents.get(x,empty) & pset)>0
        has_all_must_parents = must_parents.get(x, empty) <= pset
        if has_banned_parents or not has_all_must_parents:
            score = -np.inf
        else:
            score = scorer.score(x, ps, condtab)
        yield x, ps, score

def get_local_scores(local_scores_gen):
    local_scores = defaultdict(dict)
    for x, ps, score in local_scores_gen:
        local_scores[x][frozenset(ps)] = score
    return local_scores

def data2local_scores(valcounts, data, scorer, musts={}, bans={}):
    contabs = gen_contabs(data)
    condtabs = gen_condtabs(contabs, valcounts)
    local_scores = gen_local_scores(condtabs, scorer, musts, bans)
    return get_local_scores(local_scores)
