#!/usr/bin/env python

from typing import FrozenSet, Dict

Var = int
Varset = FrozenSet[Var]
Set2Set = Dict[Varset, Varset]
Score = float
LocalScore = Dict[Varset, Score]
LocalScores = Dict[Var, LocalScore]
Net = Dict[Var, Varset]
