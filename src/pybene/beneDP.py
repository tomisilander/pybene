#!/usr/bin/env python

from itertools import combinations
from .benetypes import *

class BeneDP():

    def __init__(self, local_scores:LocalScores):
        self.local_scores = local_scores
        self.nof_vars = len(self.local_scores)
        self.all_vars = frozenset(range(self.nof_vars))
        self.best_parents_4in = dict((x, self.get_best_parents_in_PS(x)) 
                                     for x in self.all_vars)
        self.best_net_score_in = self.get_best_net_scores_in_PS()


    def get_best_parents_in_PS(self, x) -> Set2Set :

        def get_best_parents_in_S_x_() -> Set2Set :

            def gen_scored_parents():
                for y in S_x_:
                    S__y  = S_x_ - {y}
                    parents = best_parents_in[S__y]
                    parents_score = lsx[parents]
                    yield (parents_score, parents)
                yield (lsx[S_x_], S_x_)

            return max(gen_scored_parents())[1]

        emptyset = frozenset()
        best_parents_in = {emptyset : emptyset}
        S_x = self.all_vars - {x}
        lsx = self.local_scores[x]
        for subset_size in range(1, len(S_x) + 1):
            for S_x_ in map(frozenset, combinations(S_x, subset_size)):
                best_parents_in[S_x_] = get_best_parents_in_S_x_()
        return best_parents_in


    def get_best_net_scores_in_PS(self) -> Dict[Varset, Score]:
        
        def gen_scores_for_S_():
            for x in S_:
                S__x = S_ - {x}
                x_score = ls[x][best_parents_4in[x][S__x]] 
                S__x_score = best_net_score_in[S__x]
                yield S__x_score + x_score
        
        ls = self.local_scores
        best_parents_4in = self.best_parents_4in

        emptyset = frozenset()
        best_net_score_in = {emptyset : 0.0}
        for subset_size in range(1, len(self.all_vars)+1):
            for S_ in map(frozenset, combinations(self.all_vars, subset_size)):
                best_net_score_in[S_] = max(gen_scores_for_S_())
        return best_net_score_in