#!/usr/bin/env python

from collections import defaultdict
from typing import List, Tuple, Iterable, Generator, Dict
import numpy as np
import torch
from torch import Tensor
from torch.sparse import sum as marginalize

from .scorer import Scorer


def data_mx_to_coo(data_matrix: np.ndarray, valcounts):
    """The idea is to represent the data as a sparse tensor where each entry
    corresponds to a configuration of the variables and the value at that entry
    is the count of data points with that configuration. In other words, if
    the data has n variables, then we create an n-dimensional sparse tensor where
    each dimension corresponds to a variable and the size of each dimension is
    the number of possible values for that variable (valcounts). The indices of
    the non-zero entries in the sparse tensor correspond to the configurations
    present in the data, and the values at those indices are the counts of how
    many times each configuration appears in the data.
    This is useful for efficient computation of contingency tables and
    marginalization."""

    data_matrix = np.asarray(data_matrix)
    if data_matrix.ndim != 2:
        raise ValueError("mx must be a N x D array")

    # Convert to torch long (int64) for indices
    indices_np = data_matrix.T  # shape D x N
    indices = torch.as_tensor(indices_np, dtype=torch.long)

    # Values: use float by default (match your downstream dtype if different)
    nnz = indices.shape[1]  # N
    values = torch.ones(nnz, dtype=torch.int32)

    size = tuple(int(x) for x in valcounts)

    # to get counts, we need to sum duplicates
    return torch.sparse_coo_tensor(indices, values, size).coalesce()


def read_data(filename: str, valcounts) -> Tensor:
    """reads data to a (sparse) torch tensor"""
    return data_mx_to_coo(np.loadtxt(filename, dtype=int), valcounts)


def gen_sets_down(varset: int, first_out_ix):
    """
    Go through all subsets of bitset varset by removing one variable at a time
    so that subsets are always visited after supersets. Yields (var_count, x)
    where var_count is the number of variables in the current subset
    and x is the variable removed to get to the next subset. This order is useful
    for marginalization of contab tensors.
    """

    var_count = sum(map(int, bin(varset)[2:]))

    if var_count <= 1:
        return

    for x in range(first_out_ix):
        yield (var_count, x)
        xset = 1 << x  # bitset for x
        next_set = varset ^ xset  # remove x
        yield from gen_sets_down(next_set, x)


def gen_contabs(start_contab: Tensor) -> Generator:
    """Marginalize contabs in an order dictated by gen_sets_down yielding (vars, contab)
    tuples, where vars is a tuple of variable indices remaining in contab.
    The marginalization order is from all variables down to none so that all needed
    marginals are computed only once.
    """

    n = start_contab.sparse_dim()
    contabs: List[None | Tuple[Tuple[int, ...], Tensor]] = [None] * (n + 1)
    contabs[n] = (tuple(range(n)), start_contab)
    yield contabs[n]
    start_bitset = (1 << n) - 1
    for var_count, x in gen_sets_down(start_bitset, n):
        ct_varcount = contabs[var_count]
        assert isinstance(ct_varcount, Iterable)
        old_vars, old_contab = ct_varcount
        pos_x = old_vars.index(x)
        new_vars = old_vars[:pos_x] + old_vars[pos_x + 1 :]
        new_contab: Tensor = marginalize(old_contab, [pos_x])
        contabs[var_count - 1] = (new_vars, new_contab)
        yield contabs[var_count - 1]


def contab2condtab(contab: Tensor, i: int, valcount: int):
    import numpy as np

    # indices shape (nvars, nnz) -> (nnz, nvars)
    cfgs = contab.indices().numpy().transpose().astype(np.int64)
    freqs = contab.values().numpy().astype(np.int64)
    # mask out variable i to get parent configurations
    pcfgs = cfgs.copy()
    pcfgs[:, i] = -1

    # find unique parent rows and inverse mapping
    uniq_pcfgs, inv = np.unique(pcfgs, axis=0, return_inverse=True)
    # allocate condtable: rows = number of unique parent configs
    condtab = np.zeros((uniq_pcfgs.shape[0], valcount), dtype=np.int64)

    # accumulate frequencies per (parent-row, i_value)
    # inv[j] is parent-row index for cfgs[j]; i_val = cfgs[j, i]
    np.add.at(condtab, (inv, cfgs[:, i]), freqs)

    return condtab


def gen_condtabs(contabs, valcounts):
    for varset, contab in contabs:
        for ix_out, x in enumerate(varset):
            condtab = contab2condtab(contab, ix_out, valcounts[x])
            ps = varset[:ix_out] + varset[ix_out + 1 :]
            yield x, ps, condtab


def gen_local_scores(condtabs, scorer: Scorer, 
                     must_parents: Dict = {}, banned_parents: Dict = {}
):
    empty = set()
    for x, ps, condtab in condtabs:
        pset = set(ps)
        has_banned_parents = len(banned_parents.get(x, empty) & pset) > 0
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


def data2local_scores(data: Tensor, scorer: Scorer, must_ps={}, banned_bs={}):
    contabs = gen_contabs(data)
    condtabs = gen_condtabs(contabs, scorer.valcounts)
    local_scores = gen_local_scores(condtabs, scorer, must_ps, banned_bs)
    return get_local_scores(local_scores)

