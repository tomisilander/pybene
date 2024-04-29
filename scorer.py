#!/usr/bin/env python

import numpy as np
from scipy.special import entr as _nlogn

class Scorer():

    def __init__(self, valcounts, N, score='BIC', **kwargs):
        self.set_valcounts(valcounts)
        self.kwargs = kwargs

        # some helpers

        logN = np.log(N)
        loglogN = np.log(logN)

        self.xic_penalty = {'B': lambda k: 0.5*k*logN,
                            'A': lambda k: k,
                            'H': lambda k: k*loglogN}
        
        self.score_fns = {  'BIC': self.BIC,
                            'AIC': self.AIC,
                            'HIC': self.HIC,
                        }

        self.score_fn = self.score_fns[score]

    def set_valcounts(self, valcounts):
        self.valcounts = np.asarray(valcounts)
        
    def log_ml(self, child_freqs):
        parent_freqs = child_freqs.sum(axis=1)
        res = -np.sum(_nlogn(child_freqs))   
        res += np.sum(_nlogn(parent_freqs))
        return res

    def XIC(self, X, child, parents, freqs):
        nof_pcfgs = self.valcounts[list(parents)].prod(initial=1.0)
        nof_params = nof_pcfgs * (self.valcounts[child]-1)
        return self.log_ml(freqs) - self.xic_penalty[X](nof_params)

    def BIC(self, child, parents, freqs):
        return self.XIC('B', child, parents, freqs, **self.kwargs)

    def AIC(self, child, parents, freqs):
        return self.XIC('A', child, parents, freqs, **self.kwargs)

    def HIC(self, child, parents, freqs):
        return self.XIC('H', child, parents, freqs, **self.kwargs)

    def score(self, child, parents, freqs):
        return self.score_fn(child, parents, freqs)